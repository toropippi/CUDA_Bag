import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
np.random.seed(seed=46)

programid = SourceModule("""
__global__ void ExtreamBug(double *A,double *B,int *D,int ofst,unsigned int n)
{
    double rg[16];
    int ix = threadIdx.x;

    for(int i=0;i<16;i++){
        int d=D[i];
        //ofst+i行目とd+ofst行目を入れ替えたい
        double tmp1=A[ix+(ofst+i)*n];
        double tmp2=A[ix+(d+ofst)*n];
        A[ix+(ofst+i)*n]=tmp2;
        A[ix+(d+ofst)*n]=tmp1;
        rg[i]=tmp2;
        printf("%d\\t%f\\t%d\\t%d\\n",threadIdx.x,tmp2,i,d);
        B[threadIdx.x+i*16]=tmp2;
    }

    for(int i=0;i<16;i++){
        rg[i]/=A[ofst+i+(ofst+i)*n];
    }

    for(int i=0;i<16;i++){
        A[ix+(ofst+i)*n]=rg[i];
    }
    //__syncthreads();
}
""")


n=32

#A変数
A_cpu = np.random.rand(n*n).astype(np.float64)
A_gpu = drv.mem_alloc(n*n*8)
drv.memcpy_htod(A_gpu,A_cpu)

#B変数
B_cpu = np.zeros(16*16,dtype=np.float64)
B_gpu = drv.mem_alloc(16*16*8)

#D変数
D_gpu = drv.mem_alloc(16*4)
D_cpu=np.uint32([25,30,9,16,13,30,15,7,13,23,15,27,31,18,17,18])
drv.memcpy_htod(D_gpu,D_cpu)

#カーネル実行
ExtreamBug = programid.get_function("ExtreamBug")  # 上で定義したカーネルを呼び出す
ExtreamBug(A_gpu,B_gpu,D_gpu,np.int32(0),np.uint32(n),block=(16,1,1), grid=(1,1,1))

#結果確認
drv.memcpy_dtoh(B_cpu,B_gpu)
print(B_cpu[5*16])#ここは本当は0.23495にならないといけない。0.06579が出力されたらバグ
#カーネルコードのprintをコメントアウトするか__syncthreads()のコメントアウト外すか引数のunsigned int nをint nにすると正しい結果がでる。なぜ？