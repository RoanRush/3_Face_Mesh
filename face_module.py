import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=1, min_detection_conf=0.5):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_conf
        )
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def find_face_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        faces = []
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, 
                                               self.mp_face_mesh.FACEMESH_CONTOURS,
                                               self.draw_spec, self.draw_spec)
                face = []
                for id, lm in enumerate(face_lms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([id, x, y])
                faces.append(face)
        return img, faces