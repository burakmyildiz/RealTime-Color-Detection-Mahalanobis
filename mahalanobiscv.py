import cv2
import numpy as np

def compute_mahalanobis(image, mean, inv_cov):
    differences = image - mean
    m = np.dot(differences, inv_cov)
    d_squared = np.sum(m * differences, axis=2)
    return np.sqrt(d_squared)

hsv_mean_red_lower = np.array([0, 150, 150], dtype=np.float32)
hsv_mean_red_upper = np.array([170, 150, 150], dtype=np.float32)
hsv_mean_green = np.array([60, 150, 150], dtype=np.float32)

covariance = np.array([[50, 0, 0], [0, 60, 0], [0, 0, 60]], dtype=np.float32)
inverse_cov = np.linalg.inv(covariance)

threshold_red = 3.0
threshold_green = 3.0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    md_red_lower = compute_mahalanobis(hsv, hsv_mean_red_lower, inverse_cov)
    md_red_upper = compute_mahalanobis(hsv, hsv_mean_red_upper, inverse_cov)
    md_green = compute_mahalanobis(hsv, hsv_mean_green, inverse_cov)
    
    mask_red = np.uint8(((md_red_lower < threshold_red)|(md_red_upper < threshold_red)) * 255)
    mask_green = np.uint8((md_green < threshold_green) * 255)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    
    mask_red = cv2.GaussianBlur(mask_red, (5,5), 0)
    mask_green = cv2.GaussianBlur(mask_green, (5,5), 0)
    
    contours_red, like = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, 'RED', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'GREEN', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Red Mask', mask_red)
    cv2.imshow('Green Mask', mask_green)
    cv2.imshow('Frame with Detections', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
