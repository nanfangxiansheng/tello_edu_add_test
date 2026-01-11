# 工程记录：使用tello edu无人机进行计算机视觉手势识别。

[dji-sdk/Tello-Python: This is a collection of python modules that interact with the Ryze Tello drone.](https://github.com/dji-sdk/Tello-Python)

参考的是大疆tello edu的官方支持库链接

需要git把其扒下来：

```bash
git clone https://github.com/dji-sdk/Tello-Python
```



## 环境配置：

在win11 x64系统上进行开发。基于python2.7搭建环境，由于python2.7是很老的python版本，故按照其下载文件夹中的.bat文件执行。下载后的python路径在C:\Python27\python.exe，在运行命令的时候需要加入前面的路径来执行。

其中有numpy,opencv,libh264decoder无法下载.

解决办法为：经过测试有效的，下载opencv for python2.7

```python
C:\Python27\python.exe -m pip install opencv-python==4.2.0.32
```

此外关于libh264decoder解决办法是把官方链接中的tello_video_dll(For win64).zip中的全部文件都挨个放在文件夹C:\Python27\lib\site-packages文件夹下面，如下图所示：

而不是仅仅把libh264decoder.pyd文件放在该文件夹下，.pyd文件就相当于一个动态链接库.so文件，还需要dll文件来支持。

<img width="1411" height="1218" alt="image" src="https://github.com/user-attachments/assets/6bd30c8e-722b-4951-abdf-8d1dd10e41fb" />

## 运行说明

运行的文件夹名字应当为：Tello_Video_With_Pose_Recognition1

在该文件夹下面的Model文件夹下还具有着来自Pose和yolov3的权重和配置文件：

<img width="1321" height="456" alt="image" src="https://github.com/user-attachments/assets/5301ff4b-4697-4670-b268-a01bfa653d6b" />

运行时候应当首先开机tello无人机，再打开电脑WIFI连上tello，再运行main.py在其中开启通信和主任务循环,在终端中敲入下面的指令：

```bash
C:\Python27\python.exe main.py
```

随后就进入了控制界面：

<img width="2559" height="1526" alt="image" src="https://github.com/user-attachments/assets/4f3034fc-783c-4dd2-8394-700ea4def678" />

可以看到其中集成了多个功能：open command pannel功能指向了键盘控制无人机飞行。

optical tracking status功能指向了是否开启LK光流**特征追踪**。

fast recognition指向了是否开启fast **特征提取**。

yolo recognition指向了yolo目标检测和实时bbox展现。

pose recognition指向了开启姿势检测。

## 下载测试pose模型

OpenPose人体姿态识别项目是美国[卡耐基梅隆大学](https://zhida.zhihu.com/search?content_id=10017670&content_type=Article&match_order=1&q=卡耐基梅隆大学&zhida_source=entity)（CMU）基于[卷积神经网络](https://zhida.zhihu.com/search?content_id=10017670&content_type=Article&match_order=1&q=卷积神经网络&zhida_source=entity)和监督学习并以caffe为框架开发的开源库。可以实现人体动作、面部表情、手指运动等姿态估计。适用于单人和多人，具有极好的鲁棒性。是世界上首个基于深度学习的实时多人二维姿态估计应用，基于它的实例如雨后春笋般涌现。人体姿态估计技术在体育健身、动作采集、3D试衣、舆情监测等领域具有广阔的应用前景，人们更加熟悉的应用就是抖音尬舞机。

这里应当注意的是openpose实际上对于tello python要求下载的那个模型已经在官网上不保留了，需要自己去找资源。

[GitCode - 全球开发者的开源社区,开源代码托管平台](https://gitcode.com/Resource-Bundle-Collection/de094/?utm_source=pan_gitcode&index=top&type=card&uuid_tt_dd=10_30830165890-1754646235885-646295&from_id=143385025&from_link=5eda50e38f968ae784ad1904196108c9)

在上面的链接中有下载资源。下载好后把pose_iter_160000放在model/pose/mpi中即可。

在原本的代码中要求同时读入proto file(规定了输入的神经网络的层数和每层的特点)，还有读入caffemodel参数权重文件。

```python
        # read the path of the trained model of the neural network for pose recognition
        self.protoFile = "model/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = "model/pose/mpi/pose_iter_160000.caffemodel"
        
        # total number of the skeleton nodes
        self.nPoints = 15
        
        # read the neural network of the pose recognition
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

```

下面是pose模型效果展示：

<img width="2559" height="1464" alt="image" src="https://github.com/user-attachments/assets/85f2cb28-22fd-46c7-a97b-3fce5b12fe5a" />

## 部署测试低版本Yolo

简单版本的Yolo3搭载在py2.7环境中捕获输入视频中的物体并圈出来。整体流程是点击yolo按钮后开始识别，然后处理返回的x,y,h,w在图中画出来。yolov3的模型比较难找。现在提供了一个下载链接：
https://gitcode.com/open-source-toolkit/c20d8/

<img width="1280" height="1707" alt="image" src="https://github.com/user-attachments/assets/8ba46762-6d04-45c3-8a52-ff31a8f36b86" />

 详细代码过程如下。首先是加载yolo3低版本模型的代码：

这里的逻辑是当用户按下tkinter的按钮后yolo_mode设置为True,随后加载yolo3模型的.cfg和.weights文件再获得detections.这里还隐藏了一个保存的功能，去掉taskSnapshot1的注释后可以进行保存。

```python
                if self.yolo_mode:
                    image = self.frame
                    model = cv2.dnn.readNetFromDarknet('yolo3.cfg', 'yolov3.weights')
                    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                    model.setInput(blob)
                    detections = model.forward()
                    #print(type(detections))
                    for detection in detections:
                        x, y, w, h, confidence = detection[0:5]                       
                        if confidence > 0.05:
                            height,width=image.shape[:2]
                            height=int(height)
                            width=int(width)
                            print("width=",width)
                            print("height=",height)
                            cv2.rectangle(image, (int(width * (x - w/2)), int(height * (y - h/2))), (int(width * (x + w/2)), int(height * (y + h/2))), (0, 255, 0), 2)                    
                            #takeSnapshot1(image)
                            					           print("x=",x,"y=",y,"w=",w,"h=",h,"confidence=",confidence)
                    self.detections=detections
```



下面是得到yolo返回的检测结果包含框的中心坐标x,y和宽度w,高度h，随后在打开的frame中用cv2.rectangle在图中画出框。此处应该注意返回的x,y,w,h都是归一化后的，应当乘以图像的w和h去反向归一化。

```python
            if self.yolo_mode:
                height,width=frame.shape[:2]
                height=int(height)
                width=int(width)
    
                for detection in self.detections:
                    x, y, w, h, confidence = detection[0:5]
                    #print("confidence=",confidence)
                    if confidence > 0.05:
                        cv2.rectangle(frame, (int(width * (x - w/2)), int(height * (y - h/2))), (int(width * (x + w/2)), int(height * (y + h/2))), (0, 255, 0), 2)                            
                        #print("x=",x,"y=",y,"w=",w,"h=",h)
                        #print("height=",height,"width=",width)
```

## 键盘运动控制

在tello edu的tkinter界面中同时有运动控制的一个子界面。可以实现如下的运动控制功能：

1.指定距离的前后左右移动

2.指定高度的向上运动和向下运动

3.起飞和落地

4.指定角度的顺时针旋转和逆时针旋转

<img width="762" height="513" alt="image" src="https://github.com/user-attachments/assets/7e09d286-152b-45b5-8447-cf7423fcc0ef" />

其相关代码如下：

首先是封装了向tello edu发送信息的函数。send_command(self,command)

```python
    def send_command(self, command):
        """
        Send a command to the Tello and wait for a response.

        :param command: Command to send.
        :return (str): Response from Tello.

        """

        print (">> send cmd: {}".format(command))
        self.abort_flag = False
        timer = threading.Timer(self.command_timeout, self.set_abort_flag)

        self.socket.sendto(command.encode('utf-8'), self.tello_address)

        timer.start()
        while self.response is None:
            if self.abort_flag is True:
                break
        timer.cancel()
        
        if self.response is None:
            response = 'none_response'
        else:
            response = self.response.decode('utf-8')

        self.response = None

        return response
```

后续每次向Tello无人机发送指令都可以直接把command字符串填入send_command函数来实现。比如起飞指令如下：

```python

    def takeoff(self):
        """
        Initiates take-off.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('takeoff')
```

而在ui界面控制代码中通过把按键-动作进行绑定并使得按下按键就可以执行相应的动作例如：

```python
        self.tmp_f.bind('<KeyPress-Left>', self.on_keypress_left)

```

```python
    def on_keypress_left(self, event):
        print "left %d m" % self.distance
        self.telloMoveLeft(self.distance)

```

## 基于fast的特征提取

FAST算法全名为：(Features from Accelerated Segment Test)是一个特征点提取算法的缩写。。它原理非常简单，**遍历所有的像素点，判断当前像素点是不是特征点的唯一标准就是在以当前像素点为圆心以3像素为半径画个圆（圆上有16个点），统计这16个点的像素值与圆心像素值相差比较大的点的个数。超过9个差异度很大的点那就认为圆心那个像素点是一个特征点**。

特征点提取的实际上是图像中的角点，也就是图像中颜色变化比较大的点，可以作为一个特征。

<img width="376" height="668" alt="image" src="https://github.com/user-attachments/assets/8dbbd906-c8ac-4c74-8c9b-12c9be34eadd" />

应用fast算法提取特征点的图例如上图所示。可以看出，提取出的特征点有很多在物体的边缘

fast算法的实现在opencv中集成度已经很高了，比较简单。此为其核心代码：

```python
                fast = cv2.FastFeatureDetector_create()
                keypoints = fast.detect(frame, None)
```

下面是效果展示：

<img width="1641" height="1346" alt="image" src="https://github.com/user-attachments/assets/eeca60a9-5745-4f6f-8575-7df6f2ff0524" />

## 稀疏LK光流特征追踪

光流是物体或者摄像头的运动导致的两个连续帧之间的图像对象的视觉运动的模式。它是一个向量场，每个向量是一个位移矢量，显示了从第一帧到第二帧的点的移动，如图：

<img width="435" height="194" alt="image" src="https://github.com/user-attachments/assets/f61880ab-ec82-4122-94c6-a9eb463e3ecd" />

Opencv中使用cv2.calcOpticalFlowPyrLK()函数计算一个稀疏特征集的光流，使用金字塔中的迭代 Lucas-Kanade 方法。

详细代码如下：

```python
                new_frame=frame
                feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
                lk_params = dict(winSize=(15, 15),
                                maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                
                color = np.random.randint(0, 255, (100, 3))               
                old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)#find interesting points       
                mask = np.zeros_like(old_frame)  #add mask to draw trajety

                frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)#calculate movement of interesting points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)#draw a line between new and old points
                    frame=cv2.circle(frame, (a, b), 5, color[i].tolist(), -1) #it graw circls on special points
                frame=cv2.add(frame,mask)
                old_gray=frame_gray.copy()
                p0=good_new.reshape(-1,1,2)
                old_frame=frame#after is last frame
```

逻辑是在上一帧的灰度图用goodFeaturesToTrack计算感兴趣的点并在下一帧也这样做，最后在下一帧的图中用线把前后两帧的点连接起来，这就是光流法。

下面是效果展示：

<img width="2559" height="1284" alt="image" src="https://github.com/user-attachments/assets/755d30ec-12d8-4da3-955a-757899a006be" />
