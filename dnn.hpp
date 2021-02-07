#include<iostream>
#include<cstdlib>
#include<algorithm>
#include<cmath>
using namespace std;
const int N = 1e2;

int example_size,input_size,output_size; // 样本量，输入规格，输出规格
double train_data[N][N],lable[N][N];
double *input,*output; // 本批输入输出

int DEP,WID[N]; // 网络深宽
const double rate = 0.2, discont = 0.8; // 学习率，每次反向传播占比（权重）
double W[N][N][N], b[N][N], X[N][N], Z[N][N]; // 权重矩阵(W)，偏移向量(b)，输入/输出(X)，输入的线性计算(Z=WX+b)
double dW[N][N][N],db[N][N], dX[N][N]; // 上述量对loss的总梯度（由last_d计算来）
double last_dW[N][N][N],last_db[N][N], last_dX[N][N]; // 一次反向传播计算的梯度
double tot_loss,last_loss; // 总loss，一次loss

double loss(){
    double res = 0;
    for(int i=0;i<output_size;i++)
        res += (output[i]-X[DEP][i])*(output[i]-X[DEP][i])/output_size;
    return res;
}
void dloss_dX(){ // 输出层对loss梯度
    for(int i=0;i<WID[DEP];i++)
        last_dX[DEP][i] = 2*(X[DEP][i]-output[i])/output_size;
}
double act(const double &x){return 1/(1+exp(-x));} // 激活函数
double dact(const double &x){return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));} // 激活函数导函数

namespace fp{ // 前向传播
    void cal_layer(const int &l){
        for(int i=0;i<WID[l];i++){
            double res = 0;
            for(int j=0;j<WID[l-1];j++){
                res += W[l-1][i][j] * X[l-1][j] + b[l-1][j];
            }
            Z[l][i] = res;
        }
    }
    void act_layer(const int &l){
        for(int i=0;i<WID[l];i++)
            X[l][i] = act(Z[l][i]);
    }
}
void front_propagation(){
    for(int i=0;i<WID[0];i++)
        X[0][i] = input[i];
    fp::cal_layer(0); // 只有隐藏层和输出层具有激活函数，输入层不算神经元
    for(int i=1;i<DEP;i++){
        fp::cal_layer(i); // Z_L = W_L-1 * X_L-1 + b_L-1
        fp::act_layer(i); // X_L = act(Z_L)
    }
    last_loss = loss();
}

namespace bp{ // 反向传播
    void cal_dW(const int &l){
        for(int i=0;i<WID[l+1];i++)
            for(int j=0;j<WID[l];j++)
                last_dW[l][i][j] = last_dX[l + 1][i] * dact(Z[l + 1][i]) * X[l][j];
    }
    void cal_db(const int &l){
        for(int j=0;j<WID[l];j++){
            last_db[l][j] = 0;
            for(int i=0;i<WID[l+1];i++)
                last_db[l][j] += last_dX[l + 1][i] * dact(Z[l + 1][i]);
        }
    }
    void cal_dX(const int &l){
        for(int j=0;j<WID[l];j++){
            last_dX[l][j] = 0;
            for(int i=0;i<WID[l+1];i++)
                last_dX[l][j] += last_dX[l + 1][i] * dact(Z[l + 1][i]) * dact(Z[l][j]);
        }
    }
    void update_grad(){ // 更新本次的反向传播对于总体的贡献
        for(int l=0;l<DEP;l++)
            for(int i=0;i<WID[l+1];i++)
                for(int j=0;j<WID[l];j++){
                    double &dW_ = dW[l][i][j];
                    dW_ *= discont; dW_ += last_dW[l][i][j];
                }
        for(int l=0;l<DEP;l++)
            for(int i=0;i<WID[l];i++){
                double &db_ = db[l][i], &dX_ = dX[l][i];
                db_ *= discont; db_ += last_db[l][i];
                dX_ *= discont; dX_ += last_dX[l][i];
            }
    }
}
void back_propagation(){
    dloss_dX();
    for(int i=DEP-1;i>0;i--){
        bp::cal_dW(i);
        bp::cal_db(i);
        bp::cal_dX(i);
    }
    bp::update_grad();
}
void update(){
    for(int l=0;l<DEP;l++)
        for(int i=0;i<WID[l+1];i++)
            for(int j=0;j<WID[l];j++)
                W[l][i][j] -= rate * dW[l][i][j];
    for(int l=0;l<DEP;l++)
        for(int i=0;i<WID[l];i++)
            b[l][i] -= rate * db[l][i];
}
void train(int times){
    for(int t=0; t < times; t++){
        tot_loss = 0;
        for(int i = 0;i<example_size;i++){
            input = train_data[t], output = lable[t];
            front_propagation();
            back_propagation();
            tot_loss += last_loss;
        }
        update();
        cout<<"generation "<<t<<":"<< endl;
        cout<<"the loss = "<<tot_loss<<endl;
        cout<<"*******************"<<endl;
    }
}

double rand_num(double l,double u){
    double x=rand(),y=rand();
    if(x<y)swap(x,y);
    return y/x * (u-l) + l;
}
void init_rand(){
    for(int l=0;l<DEP;l++)
        for(int i=0;i<WID[l+1];i++)
            for(int j=0;j<WID[l];j++)
                W[l][i][j] = rand_num(0,1);
    for(int l=0;l<DEP;l++)
        for(int i=0;i<WID[l];i++)
            b[l][i] = rand_num(0,1);
}