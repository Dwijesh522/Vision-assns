#------------------- Asumptions in the assignments --------------
# glyph1 is base for our car and glyph2 is stopping wall
#----------------------------------------------------------------
import cv2
import numpy as np
import sys
import glob
import math
from statistics import mean

#----------------------------------------------------------------
#------------------- global variables ---------------------------
# expecting following two images in current working directory
image_name1 = "glyph1.jpg"; image_name2 = "glyph2.jpg"
# glyph key points and descriptors
kp1=0; kp2=0; des1=0; des2=0; glyph1=0; glyph2=0; glyph_match_1 = 0; glyph_match_2 = 0; glyph_height=0; glyph_width=0
glyph_pattern_1 = 0 
glyph_pattern_2 = 0
# thresholds
MIN_MATCHES = 5
GOOD_CHESSBOEARD_IMAGES = 12
#camera calubration parameters
ret=0; mtx=0; dist=0; rvecs=0; tvecs=0; height_video_frames=0; width_video_frames=0; new_camera_matrix=0; bl_threshold = 100; white_threshold = 112; WEAK_COLOR_THRESHOLD = 50
# 3d object
object1=0; object2=0
accn = 1
v_init = 5
is_halt = False
base_middle_points = np.array([0,0])
wall_line_middle_point = np.array([0,0])
#----------------------------------------------------------------

# object class to render 3D wavefront OBJ file
# this class traverses line by line to given argument .obj file and stores data in corresponding datastructure
class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            #elif values[0] in ('usemtl', 'usemat'):
                #material = values[1]
            #elif values[0] == 'mtllib':
                #self.mtl = MTL(values[1])
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))


# given two points, find angle between two corresponding lines
def get_angle(x1, y1, x2, y2):
    slope = (y2-y1)/(x2-x1)
    return math.atan(slope)

# returns two highest value points wrt given "axis"
def get_two_max_points(points, axis):
    max1_index=0; max2_index=0; max1_value = -10000; max2_value = -10000
    if(len(points) != 4):
        print("not possible")
    elif(axis == 'y'):
        for i in range(4):
            if(points[i][1] > max1_value):
                max2_value = max1_value
                max1_value = points[i][1]
                max2_index = max1_index
                max1_index = i
            elif(points[i][1] > max2_value):
                max2_value = points[i][1]
                max2_index = i
        return (points[max1_index][0], points[max1_index][1], points[max2_index][0], points[max2_index][1])
    else:
        print("not implemented")


def modify_direction(glyph1_points, glyph2_points, image):
    # base: glyph2, wall: glyph1 
    global base_middle_points, wall_line_middle_point
    (point1_1_x, point1_1_y, point2_1_x, point2_1_y) = get_two_max_points(glyph1_points, 'y')
    wall_line_middle_point = np.array([(point1_1_x + point2_1_x)/2, (point1_1_y + point2_1_y)/2])
    base_line1_middle_point = np.array([(glyph2_points[0][0] + glyph2_points[1][0])/2, (glyph2_points[0][1] + glyph2_points[1][1])/2])
    base_line2_middle_point = np.array([(glyph2_points[2][0] + glyph2_points[3][0])/2, (glyph2_points[2][1] + glyph2_points[3][1])/2])
    base_middle_points = np.array([(base_line1_middle_point[0]+base_line2_middle_point[0])/2, (base_line1_middle_point[1] + base_line2_middle_point[1])/2])
    #cv2.line(image, (int(base_middle_points[0]), int(base_middle_points[1])), (int(wall_line_middle_point[0]), int(wall_line_middle_point[1])), (0, 0, 255), 5)
    direction = ((base_middle_points[1] - wall_line_middle_point[1])/(base_middle_points[0] - wall_line_middle_point[0]))
    return direction

# for 4th part, we need direction of motion
# given points of two rectangles, returns rotation matrix to make base line parallel
def get_rotation_matrix_between_glyphs(glyph1_points, glyph2_points, image):
    # point1 and point2 corresponds points on lines under consideration
        (point1_1_x, point1_1_y, point2_1_x, point2_1_y) = get_two_max_points(glyph1_points, 'y')
        (point1_2_x, point1_2_y, point2_2_x, point2_2_y) = get_two_max_points(glyph2_points, 'y')
    # angle of two lines under consideration wrt horizontal line
        #cv2.line(image,(point1_2_x, point1_2_y),(point2_2_x, point2_2_y), (0,0,255), 5)
        #cv2.line(image,(point1_1_x, point1_1_y),(point2_1_x, point2_1_y), (0,0,255), 5)
        ##cv2.imshow("Lag ja gale", image)
        #cv2.waitKey(10)
        angle1 = math.atan(modify_direction(glyph1_points, glyph2_points, image))
        angle2 = get_angle(point1_2_x, point1_2_y, point2_2_x, point2_2_y)
       # print("angles: ")
        #print(angle1)
        #print(angle2)
    # relative anble! don't know write now about sign and physical significance of the difference value
        relative_angle = angle1 - angle2
        angle_of_rotation=0

        # finding relative position of markers. Divided into four quadrants
        # upper quadrants
        if(point1_2_y < point2_1_y):
            angle_of_rotation = relative_angle + 180
        # lower quadrants
        elif(point1_1_y < point2_2_y):
            angle_of_rotation = relative_angle
        else:
            # need to change to appropriate value
            angle_of_rotation = relative_angle
        #print("relative angle: ")
        #print(relative_angle

        center = (width_video_frames//2, height_video_frames//2)
        #print(relative_angle)
        relative_angle = (relative_angle * 180)/3.14
        rotation_matrix = cv2.getRotationMatrix2D(center, (-1)*relative_angle, 1.0)
        #print(rotation_matrix)
        #print("Matrix printd")
        #rotation_matrix = np.array([[np.cos(angle_of_rotation), np.sin((-1)*angle_of_rotation), 0], 
    #                            [np.sin(angle_of_rotation), np.cos(angle_of_rotation), 0]])
        #base_row = np.array([0,0,1])
        #rotation_matrix = np.vstack([rotation_matrix, base_row])
        return rotation_matrix





#Get the maximum width and height to be obtained after straightening the image
def max_width_height(points):
    (tl, tr, bl ,br) = points
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(top_width), int(bottom_width))

    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    return (max_width, max_height)

def get_distance(points_marker_1, points_marker_2, image):
    global base_middle_points, wall_line_middle_point
    #print(base_middle_points)
    #print(wall_line_middle_point)
    distance = np.sqrt(((base_middle_points[0] - wall_line_middle_point[0]) ** 2) + ((base_middle_points[1] - wall_line_middle_point[1]) ** 2))
    ##cv2.line(image, (base_middle_points[0], base_middle_points[1]), (wall_line_middle_point[0], wall_line_middle_point[1]), (0, 255, 0), 5)
    left = False
    if(points_marker_1[0][0] < points_marker_2[0][0]):
        left = True
    pt_1_1 = points_marker_1[3]
    pt_1_2 = points_marker_1[2]
    #cv2.line(image, (pt_1_1[0], pt_1_1[1]), (pt_1_2[0], pt_1_2[1]), (0, 255, 0), 5)
#    if(left):
#        pt_2_1 = points_marker_2[0]
#        pt_2_2 = points_marker_2[1]
#        slope = ((pt_1_2[1] - pt_1_1[1])/(pt_1_2[0] - pt_1_1[0]))
#        c1 = pt_1_1[1] - (pt_1_1[0] * slope)
#        c2 = pt_2_1[1] - (pt_2_1[0] * ((pt_2_2[1] - pt_2_1[1])/(pt_2_2[0] - pt_2_1[0])))
#        distance = abs(c2 - c1)/(np.sqrt(1+ (slope ** 2)))
#    else:
    pt_2_1 = (2*points_marker_2[0] + points_marker_2[3])/3
    pt_2_2 = (2*points_marker_2[1] + points_marker_2[2])/3
    slope = ((pt_1_2[1] - pt_1_1[1])/(pt_1_2[0] - pt_1_1[0]))
    c1 = pt_1_1[1] - (pt_1_1[0] * slope)
    c2 = pt_2_1[1] - (pt_2_1[0] * ((pt_2_2[1] - pt_2_1[1])/(pt_2_2[0] - pt_2_1[0])))
    #distance = abs(c2 - c1)/(np.sqrt(1+ (slope ** 2)))
    
    # cv2.imshow('distance lines', image)
    # cv2.waitKey(3000)
    return (0.80 * distance)


#Marker 1 is the vertical marker whereas the Marker 2 is the horizontal marker
def move_object(points_marker_1, points_marker_2, frame_rate, image):
    global accn, v_init, is_halt
    left = False
    mark1_x = 0
    for pts in points_marker_1:
        mark1_x = mark1_x + pts[0]
    mark1_x = mark1_x/4
    mark2_x = 0
    for pts in points_marker_2:
        mark2_x = mark2_x + pts[0]
    mark2_x = mark2_x/4 
    if(mark1_x < mark2_x):
        left = True
    pt_1_1 = points_marker_1[1]
    pt_1_2 = points_marker_1[2]
    trans_matrix = 0
    dist_travelled = 0
    velocity = v_init + (accn * (1/frame_rate))
    v_init = velocity
    if(velocity <= 0):
        is_halt = True
    dist_travelled = velocity/frame_rate
    dist_slope = modify_direction(points_marker_1, points_marker_2, image)
    if(left):
        #print(points_marker_1)
        #print(points_marker_2)
        pt_2_1 = points_marker_2[0]
        pt_2_2 = points_marker_2[1]
        pt_2_3 = points_marker_2[3]
        #cv2.line(image, (pt_2_1[0], pt_2_1[1]), (pt_2_3[0], pt_2_3[1]), (255, 0, 0), 5)
        #dist_slope = ((pt_2_1[1] - pt_2_3[1])/(pt_2_1[0] - pt_2_3[0]))
        cos_theta = 1/(np.sqrt(1+ (dist_slope ** 2)))
        sin_theta = dist_slope/(np.sqrt(1+ (dist_slope ** 2)))
        x_dist = (-1) *(dist_travelled * cos_theta)
        y_dist = (-1) *(dist_travelled * sin_theta)
        trans_matrix = np.array([[x_dist, y_dist]])
    else:
        ##print("Yeh toh right hai")
        pt_2_1 = points_marker_2[2]
        pt_2_2 = points_marker_2[3]
        pt_2_3 = points_marker_2[0]
        #dist_slope = ((pt_2_2[1] - pt_2_3[1])/(pt_2_2[0] - pt_2_3[0]))
        #cv2.line(image, (pt_2_2[0], pt_2_2[1]), (pt_2_3[0], pt_2_3[1]), (255, 0, 0), 5)
        cos_theta = 1/(np.sqrt(1+ (dist_slope ** 2)))
        sin_theta = dist_slope/(np.sqrt(1+ (dist_slope ** 2)))
        x_dist = (dist_travelled * cos_theta)
        y_dist = (dist_travelled * sin_theta)
        trans_matrix = np.array([[x_dist, y_dist]])
    return (trans_matrix, dist_travelled)

def get_movement(obj, mar1_pts, mar2_pts, projection, image):
    global accn, glyph_width, glyph_height, glyph_match_1, glyph_match_2, glyph2_x, glyph2_y
    dirn = modify_direction(mar1_pts, mar2_pts, image)
    distance = get_distance(mar1_pts, mar2_pts, image)
    #print("distance is: ")
    #print(distance)
    total_travel = 0
    prev_travel = 0
    scale_matrix = np.eye(3) * 1 
    prev_trans = np.array([[0,0]])
    tot_trans = np.array([[0,0]])
    vertices = obj.vertices
    while(not is_halt):
        img = image.copy()
        (trans_matrix, dist_travelled) = move_object(mar1_pts, mar2_pts,10, img)
        tot_trans = prev_trans + trans_matrix
        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            (max_ht, max_width) = max_width_height(mar2_pts)
            pts = points.reshape(-1,1,3)
            #print(pts.shape)
            #print(projection.shape)
            dst = cv2.perspectiveTransform(pts, projection)
            rot_mtx = get_rotation_matrix_between_glyphs(mar1_pts, mar2_pts, img)
            #print(rot_mtx)
            bound_len = len(face[1])
            #print(dst.shape)
            
            #print(dst.shape)
            for idx,pt in enumerate(dst):
                temp = np.transpose(pt)
                    #print(temp.shape)
                arr = np.array([1])
                    #print(arr.shape)
                temp = np.vstack((temp,arr))
                    #print(temp.shape)
                res = np.transpose(np.dot(rot_mtx, temp))
                dst[idx] = res[0]
                #dst = cv2.warpAffine(dst,rot_mtx,(bound_len,bound_len))
            #print(dst)
            dst = np.array([[[p[0][0], p[0][1] + max_ht / 2]] for p in dst])
            #print(dst.shape)
            imgpts = np.int32(dst)
            
            for idx, pt in enumerate(imgpts):
                imgpts[idx] = np.add(imgpts[idx], tot_trans)
                #break
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
            #break        break
        cv2.imshow('Translation', img)
        cv2.waitKey(10)
        prev_trans = prev_trans + trans_matrix
        prev_travel = total_travel
        total_travel = total_travel + dist_travelled
        if(total_travel >= distance/2 and prev_travel < distance/2):
            accn = (-1) * accn
            #print("Acceleration negative")
            ##print(total_travel)

#order the points of the image in an order
def order_points(points):
    s = points.sum(axis=1)
    diff = np.diff(points, axis = 1)
    ordered_points = np.zeros((4,2), dtype = "float32")

    ordered_points[0] = points[np.argmin(s)]
    ordered_points[2] = points[np.argmax(s)]
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[3] = points[np.argmax(diff)]

    return ordered_points

#get the coordinates of the straightened image
def get_transformed_points(width, height):
    return np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype = "float32")

#Homography to form straight image from irregular image
# returns aligned image, ordred points and corresponding homography
def get_straight_quad(gray_image, src):
    src = order_points(src)

    (max_width, max_height) = max_width_height(src)
    dst = get_transformed_points(max_width, max_height)
    #Homography
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray_image, matrix, max_width_height(src))
    if(not np.linalg.det(matrix) == 0):
        matrix = np.linalg.inv(matrix)
    return (warped,src, matrix)                                                                   #### ////

#Resize the image to a specific size
def resize_image(img, new_size):
    ratio = new_size / img.shape[1]
    temp = img.shape[0] * ratio
    img = cv2.resize(img, (int(new_size), int(temp)))
    return img


#rotate image by the specified angle
def rotate_image(image, angle):
    (h,w) = image.shape[:2]
    center = (w/2,  h/2)
    rot_mtx = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mtx, (w,h))


def adjust_gamma(image, gamma=0.6):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# given a point find average intensity value
def get_avg_intensity_value(image, r, c):
	average_intensity = 0
	for i in range(5):
		for j in range(5):
			average_intensity += image[r+i-2, c+j-2]
	average_intensity /= 25
	return average_intensity	

def get_glyph_pattern(image, black_threshold, white_threshold):
 
    # collect pixel from each cell (left to right, top to bottom)
    cells = []
    image = adjust_gamma(image)
     
    cell_half_width = int(round(image.shape[1] / 10.0))
    cell_half_height = int(round(image.shape[0] / 10.0))
    ##print("dimension: ")
    ##print(cell_half_height)
    ##print(cell_half_width)
    row1 = cell_half_height*3
    row2 = cell_half_height*5
    row3 = cell_half_height*7
    col1 = cell_half_width*3
    col2 = cell_half_width*5
    col3 = cell_half_width*7

    i11 = (get_avg_intensity_value(image, row1, col1))
    i12 = (get_avg_intensity_value(image, row1, col2))
    i13 = (get_avg_intensity_value(image, row1, col3))
    i21 = (get_avg_intensity_value(image, row2, col1))
    i22 = (get_avg_intensity_value(image, row2, col2))
    i23 = (get_avg_intensity_value(image, row2, col3))
    i31 = (get_avg_intensity_value(image, row3, col1))
    i32 = (get_avg_intensity_value(image, row3, col2))
    i33 = (get_avg_intensity_value(image, row3, col3))

        # checking for glyph1 pattern
    if(     (i13 - i11 > WEAK_COLOR_THRESHOLD and i13 - i12 > WEAK_COLOR_THRESHOLD) 
       and  (i22 - i21 > WEAK_COLOR_THRESHOLD and i22 - i23 > WEAK_COLOR_THRESHOLD)
       and  (i31 - i32 > WEAK_COLOR_THRESHOLD and i33 - i32 > WEAK_COLOR_THRESHOLD)):
        cells = [0, 0, 1, 0, 1, 0, 1, 0, 1]
    elif(   (i12 - i11 > WEAK_COLOR_THRESHOLD and i12 - i13 > WEAK_COLOR_THRESHOLD)
        and (i21 - i22 > WEAK_COLOR_THRESHOLD and i23 - i22 > WEAK_COLOR_THRESHOLD)
        and (i33 - i32 > WEAK_COLOR_THRESHOLD and i33 - i31 > WEAK_COLOR_THRESHOLD)):
        cells = [0, 1, 0, 1, 0, 1, 0, 0, 1]
    else:
        cells.append(i11)
        cells.append(i12)
        cells.append(i13)
        cells.append(i11)
        cells.append(i22)
        cells.append(i23)
        cells.append(i31)
        cells.append(i32)
        cells.append(i33)
        for idx, val in enumerate(cells):
            if val > white_threshold:
                cells[idx] = 1
            else:
                cells[idx] = 0
    
    return cells
    
# #    ##print(max(image[row3, col1], image[row3, col1+1],image[row3, col1-1]))

#     # threshold pixels to either black or white
#     for idx, val in enumerate(cells):
#         if val > white_threshold:
#             cells[idx] = 1
#         else:
#             cells[idx] = 0
#     return cells

# compute 3D transformation matrix given homography and camera matrix
def projection_matrix(camera_parameters, homography):
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = np.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

# project 3D object into target frame
# given img: target frame, obj: object to project, projection: 3D projection matrix, h: height of gray glyph image, w: width of gray glyph image
def render(img, obj, projection, h, w, color=False):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 1

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        # points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        # dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        ##print(dst.shape)
        dst = np.array([[p[0][0] + w / 2, p[0][1] + h / 2] for p in dst])
#        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
#        dst[:,:,0] = rotation_matrix.dot(dst[:,:,0])
#        dst[:,:,1] = rotation_matrix.dot(dst[:,:,1])
        imgpts = np.int32(dst)
        #print("image points")
        #print(imgpts[0])
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


# reading webcam video or stored video
def process_video(video_name):
    # real time vs reading from current working directory
    if(video_name == "real_time_video"):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_name)
    num_points = 4
    SHAPE_RESIZE = 100.0
    while(True):
        # capturing frame by frame
        ret, frame = cap.read()
        ################## deleteThis ############
        frame = cv2.imread("test_case18.jpg")
        deleteThis = cv2.imread("test_case18.jpg")
#        deleteThis2 = cv2.imread("test_case3.jpg")
        ###############################################
       # frame = cv2.undistort(frame, mtx, dist, None, new_camera_matrix)
        #------------------------------------------ important parameters ------------------------------------------------
        # homography from original glyphs to target image frame from video and corresponding 3D transformation matrix
        glyph1_homography=0; glyph2_homography=0; glyph1_3d_transformation=0; glyph2_3d_transformation=0
        # boolean variable to identify which glyph is present in the target image frame
        is_glyph1_present = False; is_glyph2_present = False
        #---------------------------------------------------------------------------------------------------------------
        # getting the gray image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(gray,50,200)
#        #cv2.imshow('canny', edges)
#        #cv2.waitKey(3000)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
 #       cv2.drawContours(deleteThis, contours, -1, (0,255,0), 3)
#        cv2.imshow('contours', deleteThis)
#        cv2.waitKey(3000)
        # out of all possible contours, identify if there exist glyphs or not. If so then find corresponding homographies
        glyph_match_1=0; glyph_match_2=0
#        #print("contour size: ")
#        #print(len(contours))
        for contour in contours:
            deleteThis1 = cv2.imread("test_case18.jpg")
 #           cv2.drawContours(deleteThis1, contour, -1, (0,255,0), 3)
 #           cv2.imshow('perticular contour', deleteThis1)
  #          cv2.waitKey(3000)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
   #         #print("number of points")
   #         #print(len(approx))
            if len(approx) == num_points: 
                (fixed_quad,approx, unknown_homography) = get_straight_quad(gray,approx.reshape(4,2))               #### /////
                resized_img = resize_image(fixed_quad, SHAPE_RESIZE)
                
                kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
                # resized_img = cv2.erode(resized_img, kernel_rect1, iterations=1) 
                resized_img = cv2.medianBlur(resized_img, 5)

  #              cv2.imshow("binary", resized_img)
  #              cv2.waitKey(10)
#                if resized_img[5,5] > bl_threshold:
#                    #print("fails black edge check")
                    # if get_avg_intensity_value(resized_img, 5, 5) > bl_threshold:
#                    continue

                for i in range(4):
                    glyph_query = get_glyph_pattern(resized_img, bl_threshold, white_threshold)
                    if glyph_query == glyph_pattern_1 and (not is_glyph1_present):
     #                   #print("glyph pattern: ")
     #                   #print(glyph_query)
                        is_glyph1_present = True                                                                    #### ////
                        glyph1_homography = unknown_homography                                                      #### ////
                        glyph_match_1 = approx
                        break

                    if glyph_query == glyph_pattern_2 and (not is_glyph2_present):
     #                   #print("glyph pattern2: ")
      #                  #print(glyph_query)
                        is_glyph2_present = True                                                                    #### ////
                        glyph2_homography = unknown_homography                                                      #### ////
                        glyph_match_2 = approx
                        break

                    resized_img = rotate_image(resized_img, 90)
                    approx = np.roll(approx,1,axis = 0)
        
        # here we know if there exist glyphs in the target frame or not, if so then also the homographies
        ## calculating 3D transformatin matrix if possible
        # for first glyph
        final_frame = frame
        if(is_glyph1_present and is_glyph2_present):
    #        #print("detected two glyphs")
            glyph2_3d_transformation = projection_matrix(new_camera_matrix, glyph2_homography)
#            #print(glyph1_3d_transformation)
            #rotation_matrix = get_rotation_matrix_between_glyphs(glyph_match_1, glyph_match_2,frame)
#            #print("rotation matrix")
#            #print(rotation_matrix)
            #glyph2_3d_transformation = np.dot(rotation_matrix, glyph2_3d_transformation)
#            #print("new matrix")
#            #print(glyph1_3d_transformation)
#            final_frame = render(final_frame, object1, glyph1_3d_transformation, glyph_height, glyph_width)
            get_movement(object2,glyph_match_1,glyph_match_2, glyph2_3d_transformation, final_frame)
            #plotting lines under
            (point1_1_x, point1_1_y, point2_1_x, point2_1_y) = get_two_max_points(glyph_match_1, 'y')
            (point1_2_x, point1_2_y, point2_2_x, point2_2_y) = get_two_max_points(glyph_match_2, 'y')
            #cv2.line(final_frame, (point1_1_x, point1_1_y), (point2_1_x, point2_1_y), (0, 0, 255), 5)
            #cv2.line(final_frame, (point1_2_x, point1_2_y), (point2_2_x, point2_2_y), (0, 0, 255), 5)

        elif(is_glyph1_present):
 #           #print("glyph1 found")
            glyph1_3d_transformation = projection_matrix(new_camera_matrix, glyph1_homography)
            (max_ht, max_width) = max_width_height(glyph_match_1)
            final_frame = render(final_frame, object1, glyph1_3d_transformation, max_ht, max_width)
            # objects initialized earlier, so projecting each object point onto the target frame
            # final_frame = render(final_frame, object1, glyph1_3d_transformation, glyph_height, glyph_width)
        # for second glyph
        elif(is_glyph2_present):
            glyph2_3d_transformation = projection_matrix(new_camera_matrix, glyph2_homography)
            (max_ht, max_width) = max_width_height(glyph_match_2)
            final_frame = render(final_frame, object2, glyph2_3d_transformation, max_ht, max_width)
            # objects initialized earlier, so projecting each object point onto the target frame
            # final_frame = render(final_frame, object2, glyph2_3d_transformation, glyph_height, glyph_width)
            
        # finally showing up the video
        # cv2.imshow('augmented object video', final_frame)
        # cv2.waitKey(10000)
        break

    # when everything done release the capture
    cap.release()
    cv2.destroyAllWindows()

# glyphs initialization: find keyPoints and descriptors
def glyph_initialization():
    global glyph1, glyph2, kp1, kp2, des1, des2, glyph_height, glyph_width, glyph_pattern_1, glyph_pattern_2
    glyph1 = cv2.imread(image_name1, 0)
    glyph2 = cv2.imread(image_name2, 0)
    # to remove noice
    glyph1 = cv2.medianBlur(glyph1, 5)
    glyph2 = cv2.medianBlur(glyph2, 5)
    (glyph1_height, glyph1_width) = glyph1.shape
    (glyph2_height, glyph2_width) = glyph2.shape
    glyph_height = (glyph1_height + glyph2_height)/2
    glyph_width = (glyph1_width + glyph2_width)/2

    glyph_pattern_1 = get_glyph_pattern(glyph1, bl_threshold, white_threshold)
    glyph_pattern_2 = get_glyph_pattern(glyph2, bl_threshold, white_threshold)
    #print(glyph_pattern_1)
    #print(glyph_pattern_2)

# camera calibration
def calibration():
    global ret, mtx, dist, rvecs, tvecs, height_video_frames, width_video_frames, new_camera_matrix
    #termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #preparing object points
    objp = np.zeros((7*7, 3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
    # arrays to store object points and image points from all the images
    obj_points = []
    image_points = []
    # all the images
    images = glob.glob('*.jpg')
    # looping through images untill we get 12 good images
    good_image_count = 0
    for fname in images:
        image = cv2.imread(fname)
        #initializing height and width of frames taken by camera
        height_video_frames, width_video_frames = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # finding the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        # if found adding object points and image points
        if(ret == True):
            good_image_count += 1
            obj_points.append(objp)
            # refining the corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)
            #drawing and displaying the corners
            cornered_image = cv2.drawChessboardCorners(image, (7, 7), corners2, ret)
            # cv2.imshow("cornered_image", cornered_image)
            # cv2.waitKey(2000)
    cv2.destroyAllWindows();
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, gray.shape[::-1],None,None)
    # finding new optimal matrices
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width_video_frames, height_video_frames), 1, (width_video_frames, height_video_frames))
    # undistorting the image

# initializing the objects with .obj file
def object_initialization():
    global object1, object2
    object1 = OBJ("fox.obj", swapyz=True)
    object2 = OBJ("rat.obj", swapyz=True)

def main():
    #reading the command line argument
    arg_size = len(sys.argv)
    if(arg_size == 1):
        video_path = "real_time_video"
    else:
        video_path = sys.argv[1]
    
    calibration()
    object_initialization()
    glyph_initialization()
    process_video(video_path)

main()
