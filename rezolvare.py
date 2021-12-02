import cv2 as cv
import numpy as np
import os


def show_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def preprocess_image(image):
    image2 = image.copy()

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_m_blur = cv.medianBlur(image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    _, thresh = cv.threshold(image_sharpened, 20, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.erode(thresh, kernel)

    # show_image("median blurred", image_m_blur)
    # show_image("gaussian blurred", image_g_blur)
    # show_image("sharpened", image_sharpened)
    # show_image("threshold of blur", thresh)

# pt task 2
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    image_m_blur2 = cv.medianBlur(image2, 5)
    image_g_blur2 = cv.GaussianBlur(image_m_blur2, (0, 0), 5)
    image_sharpened2 = cv.addWeighted(image_m_blur2, 1.0, image_g_blur2, -0.883, 0)


    edges = cv.Canny(thresh, 150, 400)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if len(contours[i]) > 3:
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 500
    height = 500

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(image_copy, tuple(top_left), 4, (0, 255, 255), -1)  # galben
    cv.circle(image_copy, tuple(top_right), 4, (255, 0, 255), -1)  # mov
    cv.circle(image_copy, tuple(bottom_left), 4, (255, 255, 255), -1)  # alb
    cv.circle(image_copy, tuple(bottom_right), 4, (100, 100, 255), -1)  # portocaliu
    #     show_image("detected corners",image_copy)

    return top_left, top_right, bottom_left, bottom_right, image_sharpened2


def rotateAndCropImage(img, topLeft, topRight, botLeft, botRight):
    corners = np.array([botLeft, topLeft, topRight, botRight])
    rect = cv.minAreaRect(corners)
    latura = min(int(rect[1][0]), int(rect[1][1]))

    src_pts = corners.astype("float32")
    p1 = [0, latura - 1]
    p2 = [0, 0]
    p3 = [latura - 1, 0]
    p4 = [latura - 1, latura - 1]
    dst_pts = np.array([p1, p2, p3, p4], dtype="float32")

    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv.warpPerspective(img, M, (latura, latura))

    return warped


def compute_energy(img):
    E = np.zeros((img.shape[0], img.shape[1]))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(img_gray, ddepth=cv.CV_16S, dx=1, dy=0, borderType=cv.BORDER_CONSTANT)
    grad_y = cv.Sobel(img_gray, ddepth=cv.CV_16S, dx=0, dy=1, borderType=cv.BORDER_CONSTANT)

    abs_x = np.abs(grad_x)
    abs_y = np.abs(grad_y)
    E = abs_x + abs_y

    return E


def compute_energy2(img):
    E = np.zeros((img.shape[0], img.shape[1]))
    grad_x = cv.Sobel(img, ddepth=cv.CV_16S, dx=1, dy=0, borderType=cv.BORDER_CONSTANT)
    grad_y = cv.Sobel(img, ddepth=cv.CV_16S, dx=0, dy=1, borderType=cv.BORDER_CONSTANT)

    abs_x = np.abs(grad_x)
    abs_y = np.abs(grad_y)
    E = abs_x + abs_y

    return E


def get_results(img, lines_horizontal, lines_vertical):
    matrix = (9, 9)
    scoreMatrix = (9, 9)
    scoreMatrix = np.zeros(scoreMatrix)
    matrix = np.zeros(matrix, dtype=str)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 12
            y_max = lines_vertical[j + 1][1][0] - 12
            x_min = lines_horizontal[i][0][1] + 12
            x_max = lines_horizontal[i + 1][1][1] - 12
            patch = img[x_min:x_max, y_min:y_max].copy()
            score = cv.mean(compute_energy(patch))[0]
            scoreMatrix[i][j] = score
#             show_image("patch", patch)
#            print(score)
            if score > 130:
                matrix[i][j] = 'x'
            else: matrix[i][j] = 'o'
#     print(scoreMatrix)
    return matrix


def get_results2(img, lines_horizontal, lines_vertical):
    matrix = (9, 10)
    scoreMatrix = (9, 9)
    scoreMatrix = np.zeros(scoreMatrix)
    matrix = np.zeros(matrix, dtype=str)
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 2):
            y_min = lines_vertical[j][0][0] + 43
            y_max = lines_vertical[j + 1][1][0] + 13
            x_min = lines_horizontal[i][0][1] + 10
            x_max = lines_horizontal[i + 1][1][1] - 10
            patch = img[x_min:x_max, y_min:y_max].copy()
            score = cv.mean(compute_energy2(patch))[0]
            scoreMatrix[i][j] = score
#             if i == 0:
#                 show_image("patch", patch)
#                 print(score)
            if score > 175:
                matrix[i][j+1] = 'X'
            else: matrix[i][j+1] = 'O'
#     print(scoreMatrix)
    return matrix


def get_results3(img, lines_horizontal, lines_vertical):
    matrix = (10, 9)
    scoreMatrix = (9, 9)
    scoreMatrix = np.zeros(scoreMatrix)
    matrix = np.zeros(matrix, dtype=str)
    for i in range(len(lines_horizontal) - 2):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 10
            y_max = lines_vertical[j + 1][1][0] - 10
            x_min = lines_horizontal[i][0][1] + 40
            x_max = lines_horizontal[i + 1][1][1] + 15
            patch = img[x_min:x_max, y_min:y_max].copy()
            score = cv.mean(compute_energy2(patch))[0]
            scoreMatrix[i][j] = score
#             if i == 1:
#                 show_image("patch", patch)
#                 print(score)
            if score > 160:
                matrix[i+1][j] = 'X'
            else: matrix[i+1][j] = 'O'
#     print(scoreMatrix)
    return matrix


def showMatrix(matrix):
    for i in range(9):
        line = ''
        for j in range(9):
            line += matrix[i][j]
        print(line)


def buildColoringMatrix(matrixVertical, matrixHorizontal):
    matrix = (17, 17)
    matrix = np.zeros(matrix, dtype=str)
    for i in range(0, 16, 2):
        for j in range(0, 16, 2):
            matrix[i][j] = 'n'
            if matrixVertical[i//2][j//2 + 1] == 'X':
                matrix[i][j + 1] = '0'
            else:
                matrix[i][j + 1] = '='
            if matrixHorizontal[i//2 + 1][j//2] == 'X':
                matrix[i + 1][j] = '0'
                matrix[i + 1][j + 1] = '0'
            else:
                matrix[i + 1][j] = '='
                matrix[i + 1][j + 1] = '0'
        for i in range(0, 16, 2):
            matrix[i][16] = 'n'
            if matrixHorizontal[i//2 + 1][8] == 'X':
                 matrix[i + 1][16] = '0'
            else:
                matrix[i + 1][16] = '='
        for i in range(0, 16, 2):
            matrix[16][i] = 'n'
            if matrixVertical[8][i//2 + 1] == 'X':
                 matrix[16][i + 1] = '0'
            else:
                matrix[16][i + 1] = '='
        matrix[16][16] = 'n'
    return matrix

def fill(matrix3, width, height, x, y, start_color, color_to_update):
    if matrix3[x][y] != start_color and matrix3[x][y] != '=':
        return
    elif matrix3[x][y] == color_to_update:
        return
    else:
        matrix3[x][y] = color_to_update
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for n in neighbors:
            if 0 <= n[0] <= width - 1 and 0 <= n[1] <= height - 1:
                fill(matrix3, width, height, n[0], n[1], start_color, color_to_update)


def transformMatrix(matrix):
    original = []
    for i in range(0, 17, 2):
        l = []
        for j in range(0, 17, 2):
            l.append(matrix[i][j])
        original.append(l)
    return original


def task1(path, save_path):
    img = cv.imread(path)
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    result = preprocess_image(img)
    topLeft, topRight, botLeft, botRight, image_sharpened = result

    img_crop = rotateAndCropImage(img, topLeft, topRight, botLeft, botRight)

    w = 500
    h = 500
    dim = (w, h)
    img_crop = cv.resize(img_crop, dim, interpolation=cv.INTER_AREA)

    lines_vertical = []
    space = 55
    i = 3
    for j in range(9):
        l = []
        l.append((i, 0))
        l.append((i, h))
        lines_vertical.append(l)
        i += np.uint(space)
    lines_vertical.append([(h - 3, 0), (h - 3, h)])

    lines_horizontal = []
    space = 55
    i = 3
    for j in range(9):
        l = []
        l.append((0, i))
        l.append((w, i))
        lines_horizontal.append(l)
        i += np.uint(space)
    lines_horizontal.append([(0, w - 3), (w, w - 3)])

    energy = compute_energy(img_crop)
    energyImage = img_crop.copy()
    energyImage[:, :, 0] = energy.copy()
    energyImage[:, :, 1] = energy.copy()
    energyImage[:, :, 2] = energy.copy()

    matrix = get_results(energyImage, lines_horizontal, lines_vertical)
    showMatrix(matrix)

    file = open(save_path, "w")

    for i in range(9):
        for j in range(9):
            file.write(matrix[i][j])
        if i < 8:
            file.write('\n')

    file.close()


def task2(path, save_path):
    img = cv.imread(path)
    img = cv.resize(img, (0, 0), fx=0.2, fy=0.2)
    result = preprocess_image(img)
    topLeft, topRight, botLeft, botRight, image_sharpened = result

    img_crop = rotateAndCropImage(img, topLeft, topRight, botLeft, botRight)
    image_sharpened = rotateAndCropImage(image_sharpened, topLeft, topRight, botLeft, botRight)

    w = 500
    h = 500
    dim = (w, h)
    img_crop = cv.resize(img_crop, dim, interpolation=cv.INTER_AREA)
    image_sharpened = cv.resize(image_sharpened, dim, interpolation=cv.INTER_AREA)

    lines_vertical = []
    space = 55
    i = 3
    for j in range(9):
        l = []
        l.append((i, 0))
        l.append((i, h))
        lines_vertical.append(l)
        i += np.uint(space)
    lines_vertical.append([(h - 3, 0), (h - 3, h)])

    lines_horizontal = []
    space = 55
    i = 3
    for j in range(9):
        l = []
        l.append((0, i))
        l.append((w, i))
        lines_horizontal.append(l)
        i += np.uint(space)
    lines_horizontal.append([(0, w - 3), (w, w - 3)])

    energy = compute_energy(img_crop)
    energyImage = img_crop.copy()
    energyImage[:, :, 0] = energy.copy()
    energyImage[:, :, 1] = energy.copy()
    energyImage[:, :, 2] = energy.copy()

    energy2 = compute_energy2(image_sharpened)
    energyImage2 = image_sharpened.copy()
    energyImage2[:, :] = energy2.copy()
    energyImage2[:, :] = energy2.copy()
    energyImage2[:, :] = energy2.copy()

    matrix = get_results(energyImage, lines_horizontal, lines_vertical)
    showMatrix(matrix)

    matrix1 = get_results2(energyImage2, lines_horizontal, lines_vertical)
    for i in range(9):
        for j in range(10):
            if matrix1[i][j] == '':
                matrix1[i][j] = 'X'

    matrix2 = get_results3(energyImage2, lines_horizontal, lines_vertical)
    for i in range(10):
        for j in range(9):
            if matrix2[i][j] == '':
                matrix2[i][j] = 'X'

    matrix3 = buildColoringMatrix(matrix1, matrix2)

    color = 1
    width = len(matrix3)
    height = len(matrix3[0])

    for i in range(width):
        for j in range(height):
            if matrix3[i][j] == 'n':
                start_x = i
                start_y = j
                start_color = 'n'
                fill(matrix3, width, height, start_x, start_y, start_color, color)
                color += 1

    matrix3 = transformMatrix(matrix3)
    for i in range(len(matrix3)):
        l = []
        for j in range(len(matrix3[i])):
            l.append(matrix3[i][j])
        print(l)
        l = []
    print('\n')

    file = open(save_path, "w")

    for i in range(9):
        for j in range(9):
            file.write(matrix3[i][j] + matrix[i][j])
        if i < 8:
            file.write('\n')

    file.close()


def generatePredicitonTask(imgPath, imgSave, nrOfTests, task):
    for i in range(1, nrOfTests + 1):
        if i < 10:
            nr = '0' + str(i)
        else:
            nr = str(i)

        file_name_read = nr + ".jpg"
        imgPathFinal = os.path.join(imgPath, file_name_read)

        file_name_save = str(i) + "_predicted" + ".txt"
        file_name_save_bonus = str(i) + "_bonus_predicted" + ".txt"
        imgSaveFinal = os.path.join(imgSave, file_name_save)
        imgSaveFinalBonus = os.path.join(imgSave, file_name_save_bonus)
        fileBonus = open(imgSaveFinalBonus, "w")
        fileBonus.close()

        print("Setul", i)
        if task == 1:
            task1(imgPathFinal, imgSaveFinal)
        if task == 2:
            task2(imgPathFinal, imgSaveFinal)


#====== MAIN ======


imgPath = "D:\\sem1anul3\\AI\\Tema1\\antrenare\\clasic"
imgSave = "D:\\sem1anul3\\AI\\Tema1\\Constantinescu_Paul_332\\clasic"

imgPath2 = "D:\\sem1anul3\\AI\\Tema1\\antrenare\\jigsaw"
imgSave2 = "D:\\sem1anul3\\AI\\Tema1\\Constantinescu_Paul_332\\jigsaw"

# task1(imgPath, imgSave)
# task2(imgPath2, imgSave2)

generatePredicitonTask(imgPath, imgSave, 20, 1)
