from backend.kinova import BaseApp
import time

class Main(BaseApp):

    def start(self):
        pass
    
    def loop(self):
        pass


if __name__ == "__main__":
    final_project = Main()

    try:
        while True:
            time.sleep(0.1)  
    except KeyboardInterrupt:
        final_project.shutdown()