import numpy as np
import pickle
import sys
import json

report = False

def error(e, m):
    return np.sqrt(sum([(e[i] - m[i])**2 for i in range(len(e))]))
    #return np.sqrt(((e[0] - m[0])**2) + ((e[1] - m[1])**2))

def project_point(point, RTinv, values_dict):
    d_x, d_y, c_x0, c_y0, f = values_dict['dx'], values_dict['dy'], values_dict['xc'], values_dict['yc'], values_dict['f']

    t_cube_pt = point[:3]
    t_im_pt = point[3:]

    O_c_c = np.asarray([0, 0, 0, 1])
    O_c_w = np.matmul(RTinv, O_c_c)
    #print(O_c_w)

    # 2. get I_i^c from pixel point
    #print(c_x0, c_y0, d_x, d_y, f)
    ix = (t_im_pt[0] - c_x0) * d_x
    iy = (t_im_pt[1] - c_y0) * d_y
    I_i_c = np.asarray([ix, iy, f, 1])
    #print(I_i_c) # here things go wrong

    # 3. calculate I_i^w, i.e. pixel point in WRF
    I_i_w = np.matmul(RTinv, I_i_c)
    #print(I_i_w)

    # 4. determine t for parametric line equation
    # first, find plane of intersection; i.e. which index = 0
    for i in range(len(t_cube_pt)):
        if t_cube_pt[i] == 0:
            plane_axis = i
    #print(t_cube_pt, plane_axis)
    t = -O_c_w[plane_axis] / (I_i_w[plane_axis] - O_c_w[plane_axis])
    #print(t)
    e_pt = []
    for i in range(len(t_cube_pt)):
        if i == plane_axis:
            e_pt.append(0)
        else:
            component = O_c_w[i] + t*(I_i_w[i] - O_c_w[i])
            e_pt.append(component)
    e_pt = np.asarray(e_pt)
    #print(e_pt, t_cube_pt, e_pt - t_cube_pt)
    #print('---')
    #if report:
        #print(e_pt, t_cube_pt, e_pt - t_cube_pt)
    return e_pt

def get_pixel_error(full_transform_matrix, points):
    XYZ_w = np.asarray([p[:3] for p in points])
    extra_ones = np.ones((len(XYZ_w), 1))
    xyz_extra_ones = np.append(XYZ_w, extra_ones, axis=1)

    errors = []
    x_diff = []
    y_diff = []
    estimated_points = []
    for i, point in enumerate(xyz_extra_ones):
        result = np.matmul(full_transform_matrix, point)
        result /= result[-1]
        correct = points[i][3:]
        errors.append(error(result[:2], correct))
        x_diff.append(np.abs(result[0] - correct[0]))
        y_diff.append(np.abs(result[1] - correct[1]))
        estimated_points.append(result[:2])
        if report:
            print(x_diff[-1], y_diff[-1])
    errors = np.asarray(errors)
    pixel_error = np.mean(errors)
    #print("mean pixel error: {}\nmean x error: {}\nmean y error: {}".format(pixel_error, np.mean(x_diff), np.mean(y_diff)))
    return pixel_error, errors, estimated_points, (x_diff, y_diff)

def get_cube_error(RT, points, values_dict):
    RTinv = np.linalg.inv(RT)
    errors = []
    x_diff = []
    y_diff = []
    z_diff = []
    estimated_points = []

    for point in points:
        estimated_point = project_point(point, RTinv, values_dict)
        errors.append(error(estimated_point, point[:3]))
        diff = np.abs(estimated_point - point[:3])
        x_diff.append(diff[0])
        y_diff.append(diff[1])
        z_diff.append(diff[2])
        estimated_points.append(estimated_point)
        if report:
            #print(diff)
            print(estimated_point)
    errors = np.asarray(errors)
    cube_error = np.mean(errors)
    #print("mean cube error: {}\nmean x error: {}\nmean y error: {}\nmean z error: {}".format(cube_error, np.mean(x_diff),np.mean(y_diff), np.mean(z_diff)))
    return cube_error, errors, estimated_points, (x_diff, y_diff, z_diff)

def get_setup_values():
    get_manual = False

    try:
        jsonFile = sys.argv[1]
        #jsonFile = 'input/below-threshold-input.json'
        with open(jsonFile, encoding='utf-8') as data:
            json_data = json.loads(data.read())
        datapath = json_data['filepath']
        modelpath = 'models/' + json_data['model_name'] + '.pickle'
        #datapath = sys.argv[1]
        #modelpath = sys.argv[2]
    except:
        if 'y' in input("WARNING COULD NOT READ FILENAMES. ENTER MANUAL VALUES INSTEAD? y/n: "):
            get_manual = True
        else:
            print("STOPPING")
            sys.exit()

    if get_manual:
        datapath = input("Enter path of .CSV file: ")
        modelpath = input("Enter path of .PICKLE file: ")

    return datapath, modelpath

def estimate_k1(cx, cy, dx, dy, points, full_matrix, RT, values_dict):
    cube_error, _, _, _ = get_cube_error(RT, points, values_dict)

    # 1. calculate r_d for every point
    uv_d = np.asarray([[p[3], p[4]] for p in points])
    r_d = np.asarray([np.sqrt(np.power(p[0] - cx, 2) + np.power(p[1] - cy, 2)) for p in uv_d])

    # 2. calculate r_u for every point
    # Begin by getting the pixel errors
    pixel_error, pixel_errors, uv_u, pixel_errors_per_dimension = get_pixel_error(full_matrix, points)

    r_u = np.asarray([np.sqrt(np.power(p[0] - cx, 2) + np.power(p[1] - cy, 2)) for p in uv_u])

    # 3. estimate K_1 for each point
    K1_e = (r_u - r_d) / np.power(r_d, 3)

    # 4. pick 10 most distant points (if possible) and calculate their mean K_1
    K1_mean = np.mean(np.asarray([x for _, x in sorted(zip(r_d, K1_e), reverse=True)][:min(10, len(r_d))]))

    # 5. calculate the undistorted versions of u and v using K_1
    r_d = r_d.reshape((len(r_d), 1))
    uv_d = np.asarray(uv_d)
    uv_K = uv_d * (1 + K1_mean * np.power(r_d, 2))

    # 6. calculate the cube errors of the points estimated by K_1
    # Creating new dataset of poits in format accepted by cube error function, substituting u,v values estimated by K1 into the original u,v values
    new_k1_points = np.asarray([[points[i][0], points[i][1], points[i][2], uv_K[i][0], uv_K[i][1]] for i in range(len(r_d))])

    K1_cube_error, cube_errors, cb_e, cube_errors_per_dimension = get_cube_error(RT, new_k1_points, values_dict)

    return pixel_error, cube_error, K1_mean, K1_cube_error

def main():
    datapath, modelpath = get_setup_values()

    f = open(modelpath, 'rb')
    model_objects = pickle.load(f)
    f.close()
    full_matrix, RT, values_dict = model_objects['full_transformation_matrix'], model_objects['RT'], model_objects['values_dict']

    cx = values_dict['xc']
    cy = values_dict['yc']
    dy = values_dict['dy']
    dx = values_dict['dx']

    points = []
    with open(datapath) as file:
        for line in file:
            points.append([float(p) for p in line.split(',')])

    points = np.asarray(points)

    pixel_error, cube_error, K1, K1_cube_error = estimate_k1(cx, cy, dx, dy, points, full_matrix, RT, values_dict)

    print('---')
    print("pixel error: {:.2f} pixels\ncube error: {:.2f} mm\nK1: {:.15g}\nK1-based cube error: {:.2f} mm".format(pixel_error, cube_error, K1, K1_cube_error))

main()