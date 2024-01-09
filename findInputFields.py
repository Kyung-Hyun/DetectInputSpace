import cv2
from PIL import Image
import numpy as np
import os

def is_contour_closed(contour):
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    print(len(approx))
    return len(approx) >= 3


def calculate_overlap_percentage(region1, region2):
    # 두 부분 영역을 동일한 크기로 조절
    print("calculate--1", region1.shape[0], region2.shape[0])
    print("calculate--2", region1.shape[1], region2.shape[1])
    height = min(region1.shape[0], region2.shape[0])
    width = min(region1.shape[1], region2.shape[1])
    print(width, height)
    region1_resized = cv2.resize(region1, (width, height))
    region2_resized = cv2.resize(region2, (width, height))

    # 두 부분 영역의 겹친 영역 계산
    overlap = cv2.bitwise_and(region1_resized, region2_resized)

    # 겹친 영역 픽셀 수 계산
    overlap_area = np.sum(overlap) / 255.0

    # 두 부분 영역의 픽셀 수 계산
    area1 = np.sum(region1_resized) / 255.0
    area2 = np.sum(region2_resized) / 255.0

    # 겹침 백분율 계산
    overlap_percentage = (overlap_area / min(area1, area2)) * 100.0

    return overlap_percentage


def isSameRectangle(region1, region2):

    # 두 번째 사각형의 중심이 첫 번째 사각형의 내부에 있는지를 확인한다.
    center_x = (region2[0] + region2[2]) // 2
    center_y = (region2[1] + region2[3]) // 2

    if ((region1[0] < center_x) and (region1[2] > center_x) and
        (region1[1] < center_y) and (region1[3] > center_y)):
        return True
    else:
        return False


def get_average_brightness_of_region(image_path, left, top, right, bottom):
    # 이미지 열기
    image = Image.open(image_path)

    # 이미지를 그레이스케일로 변환
    grayscale_image = image.convert('L')

    # 부분 영역 잘라내기
    region = grayscale_image.crop((left, top, right, bottom))

    # 픽셀 데이터 가져오기
    pixels = list(region.getdata())

    # 픽셀의 평균값 계산
    average_brightness = sum(pixels) / len(pixels)

    return average_brightness


def get_max_contour_area(cropped_image):
    threshold = 225
    _, img_binary = cv2.threshold(cropped_image, threshold, 255, cv2.THRESH_BINARY_INV)

    contour_areas = []

    index = 0
    # 이미지에서 컨투어(윤곽선) 찾기
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_areas.append(cv2.contourArea(contour))
        index += 1

    return max(contour_areas)


def find_check_box_for_input(input_image, check_box_min_area=100):
    # 체크박스1 이미지 읽기
    pattern1 = cv2.imread(".//pattern//checkbox1.jpg", cv2.IMREAD_GRAYSCALE)

    # 체크박스2 이미지 읽기
    pattern2 = cv2.imread(".//pattern//checkbox2.jpg", cv2.IMREAD_GRAYSCALE)

    # 체크박스 공간 찾기
    check_boxes = []

    # 대상 이미지 읽기
    img = cv2.imread(input_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체크박스1 패턴 매칭 수행
    res = cv2.matchTemplate(img_gray, pattern1, cv2.TM_CCOEFF_NORMED)

    # 일정 유사도 이상인 부분을 찾음
    threshold = 0.7
    loc = np.where(res >= threshold)

    # 찾은 부분을 기존 찾은 부분과 중첩 여부를 확인하여 check_boxes list에 저장한다.
    for pt in zip(*loc[::-1]):
        already_detected = False

        for check_box in check_boxes:
            if (isSameRectangle(check_box, (pt[0], pt[1], pt[0] + pattern1.shape[1], pt[1] + pattern1.shape[0]))):
                already_detected = True
                break

        if (already_detected == False):
            # 매칭된 부분 이미지 내의 도형 윤곽선 검출을 이용해서 면적을 구해,
            # 작은 도형(노이즈 또는 다른 글자)이면 제외한다.
            cropped_image = img_gray[pt[1]:pt[1] + pattern1.shape[0], pt[0]:pt[0] + pattern1.shape[1]]
            if (get_max_contour_area(cropped_image) > check_box_min_area):
                check_boxes.append((pt[0], pt[1], pt[0] + pattern1.shape[1], pt[1] + pattern1.shape[0]))

    # 체크박스2 패턴 매칭 수행
    res = cv2.matchTemplate(img_gray, pattern2, cv2.TM_CCOEFF_NORMED)

    # 일정 유사도 이상인 부분을 찾음
    threshold = 0.7
    loc = np.where(res >= threshold)

    # 찾은 부분을 기존 찾은 부분과 중첩 여부를 확인하여 check_boxes list에 저장한다.
    for pt in zip(*loc[::-1]):
        already_detected = False

        for check_box in check_boxes:
            if (isSameRectangle(check_box, (pt[0], pt[1], pt[0] + pattern2.shape[1], pt[1] + pattern2.shape[0]))):
                already_detected = True
                break

        if (already_detected == False):
            # 매칭된 부분 이미지 내의 도형 윤곽선 검출을 이용해서 면적을 구해,
            # 작은 도형(노이즈 또는 다른 글자)이면 제외한다.
            cropped_image = img_gray[pt[1]:pt[1] + pattern1.shape[0], pt[0]:pt[0] + pattern1.shape[1]]
            if (get_max_contour_area(cropped_image) > check_box_min_area):
                check_boxes.append((pt[0], pt[1], pt[0] + pattern2.shape[1], pt[1] + pattern2.shape[0]))

    return check_boxes


def find_sign_box_for_input(input_image):
    # (인) 또는 서명 type1 이미지 읽기
    pattern1 = cv2.imread(".//pattern//signbox1.jpg", cv2.IMREAD_GRAYSCALE)

    # (인) 또는 서명 type2 이미지 읽기
    pattern2 = cv2.imread(".//pattern//signbox2.jpg", cv2.IMREAD_GRAYSCALE)

    # (인) 또는 서명 type3 이미지 읽기
    pattern3 = cv2.imread(".//pattern//signbox3.jpg", cv2.IMREAD_GRAYSCALE)

    # (인), 서명 공간 찾기
    sign_boxes = []

    # 대상 이미지 읽기
    img = cv2.imread(input_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (인), 서명 type1 패턴 매칭 수행
    res = cv2.matchTemplate(img_gray, pattern1, cv2.TM_CCOEFF_NORMED)

    # 일정 유사도 이상인 부분을 찾음
    threshold = 0.7
    loc = np.where(res >= threshold)

    # 찾은 부분을 기존 찾은 부분과 중첩 여부를 확인하여 sign_boxes list에 저장한다.
    for pt in zip(*loc[::-1]):
        already_detected = False

        for sign_box in sign_boxes:
            if (isSameRectangle(sign_box, (pt[0], pt[1], pt[0] + pattern1.shape[1], pt[1] + pattern1.shape[0]))):
                already_detected = True
                break

        if (already_detected == False):
            sign_boxes.append((pt[0], pt[1], pt[0] + pattern1.shape[1], pt[1] + pattern1.shape[0]))

    # (인), 서명 type2 패턴 매칭 수행
    res = cv2.matchTemplate(img_gray, pattern2, cv2.TM_CCOEFF_NORMED)

    # 일정 유사도 이상인 부분을 찾음
    threshold = 0.7
    loc = np.where(res >= threshold)

    # 찾은 부분을 기존 찾은 부분과 중첩 여부를 확인하여 sign_boxes list에 저장한다.
    for pt in zip(*loc[::-1]):
        already_detected = False

        for sign_box in sign_boxes:
            if (isSameRectangle(sign_box, (pt[0], pt[1], pt[0] + pattern2.shape[1], pt[1] + pattern2.shape[0]))):
                already_detected = True
                break

        if (already_detected == False):
            sign_boxes.append((pt[0], pt[1], pt[0] + pattern2.shape[1], pt[1] + pattern2.shape[0]))

    # (인), 서명 type3 패턴 매칭 수행
    res = cv2.matchTemplate(img_gray, pattern3, cv2.TM_CCOEFF_NORMED)

    # 일정 유사도 이상인 부분을 찾음
    threshold = 0.7
    loc = np.where(res >= threshold)

    # 찾은 부분을 기존 찾은 부분과 중첩 여부를 확인하여 sign_boxes list에 저장한다.
    for pt in zip(*loc[::-1]):
        already_detected = False

        for sign_box in sign_boxes:
            if (isSameRectangle(sign_box, (pt[0], pt[1], pt[0] + pattern3.shape[1], pt[1] + pattern3.shape[0]))):
                already_detected = True
                break

        if (already_detected == False):
            sign_boxes.append((pt[0], pt[1], pt[0] + pattern3.shape[1], pt[1] + pattern3.shape[0]))

    return sign_boxes


def find_white_space_for_input(input_image, brightness=225, max_area=3000, min_area=500, min_x=100, max_height=20, min_height=10):
    # 이미지 불러오기
    img = cv2.imread(input_image)

    # 이미지를 그레이스케일로 변환
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("temp.jpg", img_gray)

    # 이미지 이진화 (하얀색은 255, 검은색은 0)
    threshold = 225
    _, img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # 이미지에서 컨투어(윤곽선) 찾기
    contours, _ = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 정해진 크기보다 큰 문자 입력 공간 찾기
    white_spaces = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if ((area <= max_area) and (area >= min_area) and (x >= min_x) and
            (height <= max_height) and (height >= min_height) and (width > height)):

            # 픽셀의 평균값 계산
            average_brightness = get_average_brightness_of_region("temp.jpg", x, y, x + width, y + height)
            if (average_brightness > brightness):
                white_spaces.append((x, y, width, height))

    white_spaces.reverse()
    return white_spaces


def find_all_in_images(input_images, output_path):

    # 체크 박스 최소 면적
    check_box_min_area = 100

    # 입력 상자 영역의 평균 밝기 임계값(이것보다 밝은 입력 상자만 선택한다.) 
    brightness = 225

    # 입력 상자 영역의 크기 관련 경계값
    max_area = 100000       # 입력 상자 영역의 최대 면적
    min_area = 300          # 입력 상자 영역의 최소 면적
    min_x = 100             # 입력 상자 영역의 왼쪽 상단 코너 x 좌표 최소값 (왼쪽의 일정 영역은 배제하기 위함)
    max_height = 80         # 입력 상자 영역의 최대 높이
    min_height = 8          # 입력 상자 영역의 최소 높이

    for image in input_images:
        print("")
        print("###################################################")
        print("#")
        print("# Prcocessing ", image)
        print("#")
        print("###################################################")

        # 이미지 불러오기
        img = cv2.imread(image)

        ###################################################
        # 1. 체크 박스 찾기
        ###################################################
        print("##################################")
        print("# find check box for option check")
        print("##################################")

        check_boxes = find_check_box_for_input(image, check_box_min_area)
        if (len(check_boxes)):
            # 찾은 부분에 일련번호와 사각형 표시
            index = 0
            for check_box in check_boxes:
                x, y, width, height = check_box[0], check_box[1], check_box[2] - check_box[0], check_box[3] - check_box[1]
                if (index % 2):
                    cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + width, y + height), (36, 255, 12), 2)

                print("%d : (%d, %d), (%d, %d)" %(index, x, y, x + width, y + height))
                index += 1
        else:
            print("==> no check box")

        ###################################################
        # 2. 텍스트 입력 칸 찾기
        ###################################################
        print("##################################")
        print("# find white space for text input")
        print("##################################")

        text_input_spaces = find_white_space_for_input(image, brightness, max_area, min_area, min_x, max_height, min_height)
        if (len(text_input_spaces)):
            # 찾은 부분에 일련번호와 사각형 표시
            index = 0
            for text_input in text_input_spaces:
                x, y, width, height = text_input
                cv2.putText(img, str(index), (x, y + int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + width, y + height), (36, 255, 12), 2)

                # 텍스트 입력 칸 내부에 x축 방향으로 20 pixel 간격의 격자를 그린다.
                i = 0
                while (20 * i < width):
                    cv2.line(img, (x + 20 * i, y), (x + 20 * i, y + height), (0, 255, 255), 1)
                    i += 1

                # 텍스트 입력 칸 내부에 y축 방향으로 20 pixel 간격의 격자를 그린다.
                i = 0
                while (20 * i < height):
                    cv2.line(img, (x, y + 20 * i), (x + width, y + 20 * i), (0, 255, 255), 1)
                    i += 1

                print("%d : (%d, %d), (%d, %d)" %(index, x, y, x + width, y + height))
                index += 1
        else:
            print("==> no white space")

        ###################################################
        # 3. (인), 서명 박스 찾기
        ###################################################
        print("##################################")
        print("# find signature space for signing")
        print("##################################")

        sign_boxes = find_sign_box_for_input(image)
        if (len(sign_boxes)):
            # 찾은 부분에 일련번호와 사각형 표시
            index = 0
            for sign_box in sign_boxes:
                x, y, width, height = sign_box[0], sign_box[1], sign_box[2] - sign_box[0], sign_box[3] - sign_box[1]
                if (index % 2):
                    cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + width, y + height), (36, 255, 12), 2)

                print("%d : (%d, %d), (%d, %d)" %(index, x, y, x + width, y + height))
                index += 1
        else:
            print("==> no sign area")

        # 결과 이미지 저장
        file_name = os.path.basename(image)
        output_image = f".//{output_path}//result_{file_name}"
        cv2.imwrite(output_image, img)

        print("")
        print("==> %s " %output_image)
        print("")


def find_check_box_in_images(input_images, output_path):

    check_box_min_area = 100

    for image in input_images:
        check_boxes = find_check_box_for_input(image, check_box_min_area)

        # 이미지 불러오기
        img = cv2.imread(image)

        # 찾은 부분에 일련번호와 사각형 표시
        index = 0
        for check_box in check_boxes:
            x, y, width, height = check_box[0], check_box[1], check_box[2] - check_box[0], check_box[3] - check_box[1]
            if (index % 2):
                cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + width, y + height), (36, 255, 12), 2)
            print(index, ":", (x, y), (x + width, y + height))
            index += 1

        # 결과 이미지 저장
        file_name = os.path.basename(image)
        output_image = f".//{output_path}//checkbox_{file_name}"
        print(output_image)
        cv2.imwrite(output_image, img)



def find_sign_box_in_images(input_images, output_path):

    for image in input_images:
        sign_boxes = find_sign_box_for_input(image)

        # 이미지 불러오기
        img = cv2.imread(image)

        # 찾은 부분에 일련번호와 사각형 표시
        index = 0
        for sign_box in sign_boxes:
            x, y, width, height = sign_box[0], sign_box[1], sign_box[2] - sign_box[0], sign_box[3] - sign_box[1]
            if (index % 2):
                cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(img, str(index), (x - 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + width, y + height), (36, 255, 12), 2)
            print(index, ":", (x, y), (x + width, y + height))
            index += 1

        # 결과 이미지 저장
        file_name = os.path.basename(image)
        output_image = f".//{output_path}//signbox_{file_name}"
        print(output_image)
        cv2.imwrite(output_image, img)



def find_white_space_in_images(input_images, output_path):

    # 입력 상자 영역의 평균 밝기 임계값(이것보다 밝은 입력 상자만 선택한다.) 
    brightness = 225

    # 입력 상자 영역의 크기 관련 경계값
    max_area = 100000       # 입력 상자 영역의 최대 면적
    min_area = 300          # 입력 상자 영역의 최소 면적
    min_x = 100             # 입력 상자 영역의 왼쪽 상단 코너 x 좌표 최소값 (왼쪽의 일정 영역은 배제하기 위함)
    max_height = 80         # 입력 상자 영역의 최대 높이
    min_height = 8          # 입력 상자 영역의 최소 높이

    for image in input_images:
        white_spaces = find_white_space_for_input(image, brightness, max_area, min_area, min_x, max_height, min_height)

        # 이미지 불러오기
        img = cv2.imread(image)

        # 찾은 부분에 일련번호와 사각형 표시
        index = 0
        for white_space in white_spaces:
            x, y, width, height = white_space
            cv2.putText(img, str(index), (x, y + int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(img, (x, y), (x + width, y + height), (36, 255, 12), 2)
            print(index, ":", (x, y), (x + width, y + height))
            index += 1

        # 결과 이미지 저장
        file_name = os.path.basename(image)
        output_image = f".//{output_path}//whitespace_{file_name}"
        print(output_image)
        cv2.imwrite(output_image, img)


if __name__ == "__main__":
    # 처리하고자 하는 이미지 파일이 들어있는 폴더 경로 (현재 스크립트가 위치한 폴더인 경우 "./"를 사용)
    folder_path = "source_image"

    # 처리하고자 하는 이미지 파일 확장자를 지정
    extension = ".jpg"

    # 결과 이미지를 저장할 경로
    output_path = "output"

    # 폴더 내 확장자가 jpg인 모든 이미지 파일 목록을 가져온다.
    image_list = [ f".//" + folder_path + "//" + file for file in os.listdir(folder_path) if file.endswith(extension)]


    # 함수 호출
#    find_check_box_in_images(image_list, output_path)
#    find_sign_box_in_images(image_list, output_path)
#    find_white_space_in_images(image_list, output_path)

    find_all_in_images(image_list, output_path)

#    find_check_box_in_images([ ".//source_image//WJTH02_07_2.3_6.jpg"], output_path)
    print("\n********** finished **********")
