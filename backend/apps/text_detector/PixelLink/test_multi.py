import multiprocessing
import os
import random
import sys
import time
import cv2

class simulation(multiprocessing.Process):
    def __init__(self, name):
        # must call this before anything else
        multiprocessing.Process.__init__(self)

        # then any other initialization
        self.name = name
        self.number = 0.0
        sys.stdout.write('[%s] created: %f\n' % (self.name, self.number))

    def run(self):
        main()
        sys.stdout.write('[%s] running ...  process id: %s\n' 
                         % (self.name, os.getpid()))

        self.number = random.uniform(0.0, 10.0)
        sys.stdout.write('[%s] completed: %f\n' % (self.name, self.number))
def main():
    print('running training')
    time.sleep(3)
    print('end training')
if __name__=='__main__':
    for i in range(3):
        print("loop:",i)
        p = simulation(str(i))
        p.start()
        if i==2:
            cv2.waitKey()