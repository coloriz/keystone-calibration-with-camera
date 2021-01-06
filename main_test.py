from time import sleep

import cv2
import numpy as np
from pypylon import pylon


DEBUG = False


def get_inner_bounding(pts):
    tl, tr, br, bl = pts
    inner_bound = np.zeros_like(pts)
    # Top-left의 x
    inner_bound[0, 0] = max(tl[0], bl[0])
    # Top-left의 y
    inner_bound[0, 1] = max(tl[1], tr[1])
    # Bottom-right의 x
    inner_bound[2, 0] = min(tr[0], br[0])
    # Bottom-right의 y
    inner_bound[2, 1] = min(br[1], bl[1])

    inner_bound[1] = inner_bound[2, 0], inner_bound[0, 1]
    inner_bound[3] = inner_bound[0, 0], inner_bound[2, 1]

    return inner_bound


def read_one_frame_from_camera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.StartGrabbingMax(1)

    image = None

    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            # Access the image data.
            image = grab_result.Array
        grab_result.Release()
    camera.Close()

    return image


def read_one_frame_from_camera2():
    while True:
        if camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.Array
                grab_result.Release()
                return cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
            else:
                continue


def find_screen_and_projection_area(_img):
    img = _img.copy()
    if DEBUG:
        imshow_and_wait(img, 'input')

    img = cv2.bilateralFilter(img, 11, 10, 10)
    if DEBUG:
        imshow_and_wait(img, 'bilateralFilter')

    img = cv2.Canny(img, 5, 60)
    if DEBUG:
        imshow_and_wait(img, 'Canny')

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    if True:
        window_name = 'morphology'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, -3500, -300)
        cv2.imshow(window_name, cv2.pyrDown(img))
        # imshow_and_wait(img, 'morphology')
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    def is_rectangle_contour(c):
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        return len(approx) == 4

    contours = list(filter(is_rectangle_contour, contours))

    if DEBUG:
        print(f'len(contours) : {len(contours)}')
        for i, c in enumerate(contours[:5]):
            draw_c = cv2.drawContours(_img.copy(), [c], -1, (0, 255, 0), 2)
            imshow_and_wait(draw_c, f'contour {i}')
    # contours = contours[::2]
    screen, projection_area = contours[:2]
    screen = cv2.approxPolyDP(screen, 0.1 * cv2.arcLength(screen, True), True)
    projection_area = cv2.approxPolyDP(projection_area, 0.1 * cv2.arcLength(projection_area, True), True)

    if True:
        sc = _img.copy()
        cv2.drawContours(sc, [screen], -1, (0, 255, 0), 4)
        cv2.drawContours(sc, [projection_area], -1, (0, 0, 255), 4)
        window_name = 'screen, projection'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, -3500, -1000)
        cv2.imshow(window_name, cv2.pyrDown(sc))
        # imshow_and_wait(sc, 'screen, projection')

    return screen, projection_area


def imshow_and_wait(_img, text=None):
    img = _img.copy()
    if text:
        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    window_name = 'imshow_and_wait'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, -3500, -1000)
    cv2.imshow(window_name, img)
    cv2.waitKey()


def align_rect_points(pts):
    rect = np.zeros((4, 2), dtype=pts.dtype)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def find_intersection(o1, p1, o2, p2):
    """Finds the intersection of two lines.
    The lines are defined by (o1, p1) and (o2, p2)"""
    x = o2 - o1
    d1 = p1 - o1
    d2 = p2 - o2

    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return None

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    return o1 + d1 * t1


def find_desired_rect(projection_area, shape):
    # desired rectangle의 중심이 될 점 : projection area 사선의 중심
    center_point = find_intersection(projection_area[0], projection_area[2], projection_area[1], projection_area[3])
    projection_area_contour = projection_area.reshape(-1, 1, 2)
    display_ratio = DISPLAY_WIDTH / DISPLAY_HEIGHT
    projection_area_bw = cv2.drawContours(np.zeros(shape[:2], dtype=np.uint8), [projection_area_contour], -1, 255, -1)

    def is_rect_in_projectoin_area(rect):
        return all(map(lambda p: cv2.pointPolygonTest(projection_area_contour, tuple(p), False) > 0, rect))

    def create_rectangle(center_point, i):
        p1 = center_point[0] - display_ratio * i, center_point[1] - i
        p2 = center_point[0] + display_ratio * i, center_point[1] - i
        p3 = center_point[0] + display_ratio * i, center_point[1] + i
        p4 = center_point[0] - display_ratio * i, center_point[1] + i
        return np.array([p1, p2, p3, p4], np.int32)

    def maximum_i(center_point, initial_i):
        i = initial_i
        while True:
            rect = create_rectangle(center_point, i)

            if False:
                debug_draw = cv2.drawContours(projection_area_bw.copy(), [rect.reshape(-1, 1, 2)], -1, 127, -1)
                cv2.imshow('combined', debug_draw)
                cv2.waitKey()

            if is_rect_in_projectoin_area(rect):
                i += 1
                continue
            break

        return i - 1

    i = 1
    i = maximum_i(center_point, i)
    all_failed = True
    while True:
        # move center to left
        # print('Moving left')
        center_point_backup = center_point.copy()
        i_backup = i
        center_point[0] -= 2
        i = maximum_i(center_point, i)
        if i <= i_backup:
            # print('Moving left failed')
            center_point = center_point_backup
            i = i_backup
        else:
            all_failed = False

        # move center to right
        # print('Moving right')
        center_point_backup = center_point.copy()
        i_backup = i
        center_point[0] += 2
        i = maximum_i(center_point, i)
        if i <= i_backup:
            # print('Moving right failed')
            center_point = center_point_backup
            i = i_backup
        else:
            all_failed = False

        # move center to up
        # print('Moving up')
        center_point_backup = center_point.copy()
        i_backup = i
        center_point[1] -= 2
        i = maximum_i(center_point, i)
        if i <= i_backup:
            # print('Moving up failed')
            center_point = center_point_backup
            i = i_backup
        else:
            all_failed = False

        # move center to down
        # print('Moving down')
        center_point_backup = center_point.copy()
        i_backup = i
        center_point[1] += 2
        i = maximum_i(center_point, i)
        if i <= i_backup:
            # print('Moving down failed')
            center_point = center_point_backup
            i = i_backup
        else:
            all_failed = False

        # print(f'all failed? = {all_failed}')

        if all_failed:
            break
        else:
            all_failed = True

    i = maximum_i(center_point, i)
    desired_rect = create_rectangle(center_point, i)

    return desired_rect


def find_screen_and_projection_area2():
    # white_bg = 255 * np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), np.uint8)
    # cv2.imshow(WINDOW_NAME, white_bg)
    # cv2.waitKey(100)
    distorted = read_one_frame_from_camera2()
    screen, projection_area = find_screen_and_projection_area(distorted)
    return distorted, screen, projection_area


def get_transformation_matrix():
    distorted, screen, projection_area = find_screen_and_projection_area2()
    projection_area = align_rect_points(projection_area.reshape((4, 2)))
    desired_rect = find_desired_rect(projection_area, distorted.shape)

    # projection area와 출력할 이미지의 크기를 일치시킴
    norm_pa = np.zeros(projection_area.shape, np.float32)
    min_pa_x = np.min(projection_area[:, 0])
    min_pa_y = np.min(projection_area[:, 1])
    ptp_pa_x = np.ptp(projection_area[:, 0])
    ptp_pa_y = np.ptp(projection_area[:, 1])
    norm_pa[:, 0] = (projection_area[:, 0] - min_pa_x) / ptp_pa_x * DISPLAY_WIDTH
    norm_pa[:, 1] = (projection_area[:, 1] - min_pa_y) / ptp_pa_y * DISPLAY_HEIGHT

    image_rect = np.array([[0, 0], [DISPLAY_WIDTH, 0], [DISPLAY_WIDTH, DISPLAY_HEIGHT], [0, DISPLAY_HEIGHT]], np.float32)

    # image와 projection 공간 사이의 변환 매트릭스들
    m_projection_to_image = cv2.getPerspectiveTransform(norm_pa, image_rect)

    # image 크기와 맞춘 desired rect
    norm_dr = np.zeros(desired_rect.shape, np.float32)
    norm_dr[:, 0] = (desired_rect[:, 0] - min_pa_x) / ptp_pa_x * DISPLAY_WIDTH
    norm_dr[:, 1] = (desired_rect[:, 1] - min_pa_y) / ptp_pa_y * DISPLAY_HEIGHT

    # projection 공간 상의 desired rect를 digital 공간의 좌표로 변환
    dr_on_digital = cv2.perspectiveTransform(norm_dr.reshape(-1, 1, 2), m_projection_to_image)
    # 최종적으로 image를 digital 공간 상의 좌표로 변환한 desired rect로 변환하는 매트릭스를 구함
    m_image_to_desired = cv2.getPerspectiveTransform(image_rect, dr_on_digital)

    return m_image_to_desired


def imshow_to_screen(_img, delay=1):
    global switch
    img = _img.copy()
    cv2.rectangle(img, (0, 0), (DISPLAY_WIDTH - 2, DISPLAY_HEIGHT - 2), (255, 255, 255), 10)
    cv2.imshow(WINDOW_NAME, img)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        return False
    elif key == ord('t'):
        switch = True if not switch else False
    return True


WINDOW_NAME = 'window'
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 1920, 0)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
switch = False

img_yosemite = cv2.imread('images/sku.jpg')
imshow_to_screen(img_yosemite)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.MaxNumBuffer = 2
camera.StartGrabbing()

while True:
    M = get_transformation_matrix()
    if switch:
        y = cv2.warpPerspective(img_yosemite, M, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    else:
        y = img_yosemite.copy()
    if not imshow_to_screen(y, 33):
        break

camera.StopGrabbing()
camera.Close()
