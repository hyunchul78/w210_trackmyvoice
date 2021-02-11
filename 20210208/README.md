# W210 Capstonr

## Code list

### Master code

: Master code bridges message between Matrix-odas code and DeepSpeech. USB detection and DB synchronization will be implemented later

- master/server.cpp

```c++
// line 31-32, Queueing message from matrix-odas code and DeepSpeech
std::queue<std::string> m_queue1;
std::queue<std::string> m_queue2;

// line 61~65, Creating message send/receive thread
std::thread ds_t_send(senda, c_sock1, std::ref(m_queue2), std::ref(m2));
std::thread ds_t_recv(recva, c_sock1, std::ref(m_queue1), std::ref(m1));

std::thread mv_t_send(senda, c_sock2, std::ref(m_queue1), std::ref(m1));
std::thread mv_t_recv(recva, c_sock2, std::ref(m_queue2), std::ref(m2));

// line 79, Message receive thread : it receives message from matrix-odas code and DeepSpeech code
void recva(int sock, std::queue<std::string> &m_queue, std::mutex &m)

// line 97, Message send thread : it delivers message from matrix-odas and DeepSpeech code to each other
void senda(int sock, std::queue<std::string> &m_queue, std::mutex &m)
```



### Matrix-odas code

- matrix/matrix-odas.cpp

```c++
void angle_calculate_thread(hal::MatrixIOBus &bus, hal::EverloopImage &image1d, hal::Everloop &everloop)
/* line 183, Angle computation thread: it computes angle and store it based on sound source coordinates from ODAS  */

void recva(int sock, std::queue<std::string> &m_queue, std::mutex &m)
/* line 340, Message receive thread from Master  */

void senda(int sock, std::queue<std::string> &m_queue, hal::GPIOControl &gpio, std::mutex &m)
/* line 373, Command thread that works based on message from Master code
| 1. if message is  "Detection" (Detection means Voice is detected):
|      copy angle data to Angle buffer
| 2. if message is  "Trigger" (Trigger means detected voice is trigger voice):
|     Rotate motor by the amount angle stored in Step 1 and send "Finish Rotate" message to Master.
*/
```



### Angle buffer Class

: Object that stores Angle

- matrix/angle_buffer.h
- matrix/angle_buffer.cpp

```c++
// In angle_buffer.cpp
void angle_buffer::push_buffer(int angle){
    /* If the received angle value is different from the previously received value, save the received angle to buffer */

    if(buffer_len ==0){
	angles.push(angle);
	buffer_len++;
	count++;
	std::cout << "pushed: " << angle << std::endl;
	return;
    }
    if(angle == angles.back()){
        return;
    }
    else{
        angles.push(angle);
	buffer_len++;
	count++;
	if(buffer_len > 10){
	    angles.pop();
	    buffer_len--;
	}
	//std::cout << "In Buffer: ";
	//show_elements(angles);
	//std::cout << std::endl;
    }
}
```



### DeepSpeech

: Running STT (Speech-To-Text) in real time and send message to Master code if Voice is detected or match Trigger text

- deepSpeech/streaming_sst.py

```python
# line 199, Send message when voice is detected
sendData = str("Detection")

# line 207~216, check the speech-to-text match Trigger text(such as Hi Friend) or not. If Trigger text is detected, send "Trigger" message
text = stream_context.finishStream()
print("Recognized: %s" % text)
if ('f' in text) and ('r' in text) :
    tc = str("Trigger")
    clientSock.send(tc.encode())
    recvData = clientSock.recv(1024)
    print("Finish Rotate")
else:
    tc = str("0")
    print("0")
```



# Execution instruction

## 1. ODAS Setup [LINK](https://github.com/matrix-io/odas/tree/master/demo/matrix-demos)

```
# Add repo and key
curl https://apt.matrix.one/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.matrix.one/raspbian $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/matrixlabs.list

# Update packages and install
sudo apt-get update
sudo apt-get upgrade

# Installation
sudo apt install matrixio-creator-init
sudo apt install libmatrixio-creator-hal
sudo apt install libmatrixio-creator-hal-dev
sudo reboot

sudo apt install matrixio-kernel-modules
sudo reboot

sudo apt-get install g++ git cmake
sudo apt-get install libfftw3-dev
sudo apt-get install libconfig-dev
sudo apt-get install libasound2-dev
sudo apt install libjson-c-dev

cd odas
mkdir build
cd build
cmake ..
make
```



## 2. DeepSpeech Setup [LINK](https://github.com/touchgadget/DeepSpeech)

```
sudo apt install git python3-pip python3-scipy python3-numpy python3-pyaudio libatlas3-base
pip3 install deepspeech --upgrade
pip3 install halo webrtcvad --upgrade

cd deepSpeech
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```



## 3. Matrix-odas code build

```
$ cd matrix
$ make
```



## 4. Master code build

```
$ cd master
$ make
```



## 5. code execution script : It launches all the codes

```
$ ./svrt.sh
```
