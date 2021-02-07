//
// Created by Administrator on 2021/2/6.
//
#include "dnn.hpp"
void output_net_value(){
    for(int l=0;l<DEP;l++){
        cout<<"in layer "<<l<<":"<<endl;
        cout<<"the matrix W is:"<<endl;
        for(int i=0;i<WID[l+1];i++){
            cout<<"\t[";
            for(int j=0;j<WID[l];j++)
                cout<<W[l][i][j]<<", ";
            cout<<"]"<<endl;
        }
        cout<<"the vector b is:"<<endl;
        cout<<"\t[";

        for(int i=0;i<WID[l];i++)
            cout<<b[l][i]<<", ";
        cout<<"]"<<endl;
        cout<<"*****************"<<endl;
    }
};
int main() {

    DEP = 2;
    WID[0] = 2, WID[1] = 2, WID[2] = 1;
    example_size = 4;
    input_size = 2;
    output_size = 1;

    train_data[0][0] = 0;train_data[0][1] = 0;lable[0][0] = 0;
    train_data[1][0] = 0;train_data[1][1] = 1;lable[1][0] = 1;
    train_data[2][0] = 1;train_data[2][1] = 0;lable[2][0] = 1;
    train_data[3][0] = 1;train_data[3][1] = 1;lable[3][0] = 0;

    init_rand();
    train(200);
    output_net_value();
    return 0;
}