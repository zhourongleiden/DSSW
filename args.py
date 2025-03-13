import argparse

def args():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--dataname',
                        type=str,
                        default='LargeKitchenAppliances')
    
    parser.add_argument("--noc",
                        type=int, 
                        default=1)

    parser.add_argument("--w1",
                        type=float, 
                        default=0.333)  

    parser.add_argument("--w2",
                        type=float, 
                        default=0.333)  

    parser.add_argument("--alg",
                        type=str, 
                        default="DSSW")     
    
    args = parser.parse_args()
    args.csv_data_TRAIN_path = './data/'+args.dataname+'/'+args.dataname+'_TRAIN'
    args.csv_data_TEST_path = './data/'+args.dataname+'/'+args.dataname+'_TEST'
    return args


if __name__ == "__main__":  
    arguments = args()


    
    
