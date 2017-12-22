import numpy as np

filepath = 'all_soda_labelledManually.txt'
dict = ['o','b_site','i_site','b_device','i_device','b_sensor','i_sensor','b_room','i_room']
with open(filepath,'r') as fp:
    line = fp.readline()
    cnt = 1
    data = np.array([1,2,2])
    room_flag = False
    letter_count = 0
    for i in line[3:5]:
        if not i.isdigit():
            if i == '_':
                data = np.append(data, 0)
            else:
                data = np.append(data, 3)
        if i.isdigit():
            data = np.append(data, 4)
    if line[5].isdigit():
        room_flag = True
        room_start_number = True
    for i in line[5:13]:
       if not i.isdigit():
           if i == '_':
               if letter_count == 1:
                   data[-1] = data[-2]
               data = np.append(data, [0])
           if (i == 'R') & (letter_count < 2):
               data = np.append(data, [7])
               room_flag = True
               letter_count = letter_count + 1
           elif (i != 'R') & (letter_count < 2):
               data = np.append(data, [3])
               room_flag = False
               letter_count = letter_count + 1
           elif letter_count == 2:
               data[-1] = 5
               data = np.append(data, [6])
               room_flag = False
               letter_count = letter_count + 1
           elif letter_count > 2:
               data = np.append(data, [6])
               room_flag = False
               letter_count = letter_count + 1
       if i.isdigit():
           if room_flag:
               if room_start_number:
                   data = np.append(data, 7)
                   room_start_number = False
               else:
                   data = np.append(data, 8)
           else:
               data = np.append(data, 4)
           letter_count = 0
    print(data)
    output = data
    while line:

        line = fp.readline()

        data = np.array([1, 2, 2])
        room_flag = False
        letter_count = 0
        for i in line[3:5]:
            if not i.isdigit():
                if i == '_':
                    data = np.append(data, 0)
                else:
                    data = np.append(data, 3)
            if i.isdigit():
                data = np.append(data, 4)
        if line[5].isdigit():
            room_flag = True
            room_start_number = True
        for i in line[5:13]:
            if not i.isdigit():
                if i == '_':
                    if letter_count == 1:
                        data[-1] = data[-2]
                    data = np.append(data, [0])
                if (i == 'R') & (letter_count < 2):
                    data = np.append(data, [7])
                    room_flag = True
                    letter_count = letter_count + 1
                elif (i != 'R') & (letter_count < 2):
                    data = np.append(data, [3])
                    room_flag = False
                    letter_count = letter_count + 1
                elif letter_count == 2:
                    data[-1] = 5
                    data = np.append(data, [6])
                    room_flag = False
                    letter_count = letter_count + 1
                elif letter_count > 2:
                    data = np.append(data, [6])
                    room_flag = False
                    letter_count = letter_count + 1
            if i.isdigit():
                if room_flag:
                    if room_start_number:
                        data = np.append(data, 7)
                        room_start_number = False
                    else:
                        data = np.append(data, 8)
                else:
                    data = np.append(data, 4)
                letter_count = 0
        output = np.append(output,data,axis=0)
        print("Line {}: {}".format(cnt, data))
        cnt += 1
fp.close()