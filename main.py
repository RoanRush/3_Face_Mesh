import cv2
from face_module import FaceMeshDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1) # Mirror for natural feel
        img, faces = detector.find_face_mesh(img)
        
        if len(faces) != 0:
            # Point 10 is the forehead, Point 152 is the chin
            print(f"Detected {len(faces)} face(s). Tracking 468 points.")

        cv2.imshow("RTX 5060 Face Mesh", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()