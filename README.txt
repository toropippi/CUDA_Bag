実行環境
GPU RTX2080Ti (GTX1080でも再現)
CPU core i9 9920X
OS windows 10 home

Cuda compilation tools, release 10.1, V10.1.168
python 3.7.1 (Anaconda)
pyCUDA 2019.1 (CUDA 10.1に対応,pipコマンドでinstall)



やりたいこと
・正方行列Aがある。その行のswapをGPUでやりたい
・Aは32*32の形で、swapしたい領域はその左半分(16*32)。
・D[0]が25なので0行目と25行目を入れ替えたい。これを逐次D[15]まで実行したい
・バグコードでこれを進めていくとD[4]=13のところで13行目と4行目が入れ替わったはずなのに、D[8]=13の時点で13行目には4行目のデータが入ってないということが起こる