# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:32:15 2018

@author: Dell
"""

import time
import threading

run_sig = True

class Test(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.notif = [True]
        self.name = name
        
    def run(self):
        print('Starting')
        count_time(self.name, self.notif, 5)
        print('Stoping')
        
        
def count_time(thread_name, notifier, counter):
    while counter>0:
        if not run_sig:#notifier[0]:
            thread_name.exit()
            print('stopped')
            break
        print(counter)
        time.sleep(1)
        counter -= 1

test = Test('Thread-1')
test.start()
time.sleep(2)
print('im here now')
run_sig = False

#
#
#exitFlag = 0
#
#class myThread (threading.Thread):
#   def __init__(self, name):
#      threading.Thread.__init__(self)
##      self.threadID = threadID
#      self.name = name
##      self.counter = counter
#   def run(self):
#      print ("Starting " + self.name)
#      print_time(self.name, 5, 1)
#      print ("Exiting " + self.name)
#
#def print_time(threadName, counter, delay):
#   while counter:
#      if exitFlag:
#         threadName.exit()
#      time.sleep(delay)
#      print (threadName, time.ctime(time.time()))
#      counter -= 1
#
## Create new threads
#thread1 = myThread("Thread-1")
#thread2 = myThread("Thread-2")
#
## Start new Threads
#thread1.start()
#thread2.start()
#
#print ("Exiting Main Thread")