import collections
import myo
import time
import sys
import csv
alllei = ['double', 'fist', 'spread', 'six', 'wavein', 'waveout', 'yes', 'no']

# 设置全局参数,主要是:人名和动作标签
motion_Label = 0
peoplename = 'zhouxufeng'

csvfile = open('../myodata/actdata/'+peoplename+'_'+alllei[motion_Label]+'_a.txt', "a", newline='')
writer = csv.writer(csvfile)
b_filename = '../myodata/actdata/'+peoplename+'_'+alllei[motion_Label]+'_b.txt'
c_filename = '../myodata/actdata/'+peoplename+'_'+alllei[motion_Label]+'_c.txt'

def Matrix_to_CSV(filename, data):
    with open(filename, "a", newline='', ) as csvfiles:
        writers = csv.writer(csvfiles)
        for row in data:
            writers.writerow(row)

# def Matrix_to_CSV(filename, data):
#     with open(filename, "a", newline='', ) as csvfiles:
#         writers = csv.writer(csvfiles)
#         writers.writerow([row])

class EmgDataRecode(myo.DeviceListener):
    def __init__(self, n_forRate):
        super(EmgDataRecode, self).__init__()
        # time/last_time/n for rate recode
        self.times = collections.deque()
        self.last_time = None
        self.n = int(n_forRate)
        self.__slideWindow = collections.deque(maxlen=50)
        self.activeEMG = collections.deque()
        self.active_NUM = 0
        self.onrecdoe = False
        self.unrelax = False
        self.tmpslide = 0.0  # just for print check:see the
        self.tmplen = 0  # just for print check
        self.tt = None  # rate calculated by slideWindow
        self.tt_l = 0.0


    @property
    def rate(self):
        if not self.times:
            return 0.0
        else:
            return 1.0 / (sum(self.times) / float(self.n))

    def on_connected(self, event):
        print("Hello, '{}'! Double tap to exit.".format(event.device_name))
        event.device.stream_emg(True)

    def on_emg(self, event):
        self.__emg = event.emg
        # print(event.emg)
        writer.writerow(self.__emg)
        self.__slideWindow.append(sum(list(map(abs, self.__emg))))

        # for calculate Rate
        t = time.clock()
        if self.last_time is not None:
            self.times.append(t - self.last_time)
            if len(self.times) > self.n:
                self.times.popleft()
        self.last_time = t

        self.tmpslide = sum(self.__slideWindow)/50.0
        self.onrecdoe = True if(sum(self.__slideWindow)/50.0 > 100.0) else False

        # 根据状态进行记录
        # active结束了,但是unrelax还未结束,表示该动作截止.
        if self.onrecdoe:
            self.activeEMG.append(self.__emg)
            if not self.unrelax:
                self.tt = time.clock()
        elif self.unrelax:  # 记录结束,写入结果并及时改写unrelax和清空activeEMG
            self.unrelax = False
            print()
            if len(self.activeEMG) > 60:
                Matrix_to_CSV(b_filename, self.activeEMG)
                Matrix_to_CSV(c_filename, [[len(self.activeEMG)]])
                self.active_NUM += 1
                print('act length:', len(self.activeEMG), '\tEMG Rate:', len(self.activeEMG)/(time.clock()-self.tt))
            self.activeEMG.clear()
            self.tt = None
        self.unrelax = True if self.onrecdoe else False


def main():
    myo.init(sdk_path='./myo-sdk-win-0.9.0/')
    hub = myo.Hub()
    listener = EmgDataRecode(n_forRate=50)
    while hub.run(listener.on_event, 500):
        print("\rEMG Rate:", listener.rate, 'act Num:', listener.active_NUM, 'slide mean:', listener.tmpslide, end='')
        sys.stdout.flush()
        if listener.active_NUM >= 10:
            break
    print("\n\033[1;32;mYou have finish this act.\nPlease have a rest!")


if __name__ == '__main__':
    main()
