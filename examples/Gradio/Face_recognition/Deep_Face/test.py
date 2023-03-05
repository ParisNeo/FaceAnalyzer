import cv2
import gradio as gr

class WebcamRecorder:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.recording = False
        self.out = None
        self.filename = 'output.avi'
        
    def record(self):
        self.recording = not self.recording
        if self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_width = int(self.video_capture.get(3))
            frame_height = int(self.video_capture.get(4))
            self.out = cv2.VideoWriter(self.filename, fourcc, 20.0, (frame_width, frame_height))
            
        while self.recording:
            ret, frame = self.video_capture.read()
            if ret:
                self.out.write(frame)
            else:
                break
        self.out.release()
        self.video_capture.release()

    def get_output(self):
        def interface_fn():
            record_button = gr.Button(label="Record")
            return gr.Interface(
                fn=self.record,
                inputs=[record_button],
                outputs=["webcam"],
                title="Webcam Recorder",
                description="Press the 'Record' button to start recording the webcam stream. Press the 'Record' button again to stop recording.",
                theme="default",
                layout="vertical",
                show_input=True,
                show_output=True,
                live=True,
                examples=[]
            )
        return interface_fn

webcam_recorder = WebcamRecorder()
output_interface = webcam_recorder.get_output()()
output_interface.launch()
