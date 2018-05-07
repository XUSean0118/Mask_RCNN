# coding=UTF-8
import numpy as np
import argparse
import h5py

exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
index = [0, 3, 8, 1, 2, 80, 6, 4, 7]
def printname(name):
    print(name)

def main():
    f = h5py.File(FLAGS.input, mode='r')
    h5f = h5py.File('tmp.h5', 'w')
    #print(f.attrs['layer_names'])
    h5f.attrs['layer_names'] =f.attrs['layer_names']
    for name in f.attrs['layer_names']:
        g = f[name]
        grp = h5f.create_group(name)

        grp.attrs["weight_names"] = g.attrs['weight_names']
        #print(grp.attrs["weight_names"])
        
        for weight_name in g.attrs['weight_names']:
            weight_values = g[weight_name]

            if name.decode("utf8") == "mrcnn_bbox_fc":
                size = weight_values.shape[:-1]
                val = weight_values.value
                indces = []
                for x in index:
                    for i in range(4):
                        indces.append(x*4+i)
                if size: 
                    val[:,-4:] = np.random.normal(0, np.sqrt(2.0/1024/4), (1024,4))
                    grp.create_dataset(name=weight_name, data=val[:,indces])
                else:
                    val[-4:] = 0.0
                    grp.create_dataset(name=weight_name, data=val[indces])

            elif name.decode("utf8") == "mrcnn_class_logits":
                size = weight_values.shape[:-1]
                val = weight_values.value
                if size: 
                    val[:,-1] = np.random.normal(0, np.sqrt(2.0/1024), (1024))
                    grp.create_dataset(name=weight_name, data=val[:,index])
                else:
                    val[-1] = 0.0
                    grp.create_dataset(name=weight_name, data=val[index])

            elif name.decode("utf8") == "mrcnn_mask":
                size = weight_values.shape[:-1]
                val = weight_values.value
                if size: 
                    val[:,:,:,-1] = np.random.normal(0, np.sqrt(2.0/256), (1,1,256))
                    grp.create_dataset(name=weight_name, data=val[:,:,:,index])
                else:
                    val[-1] = 0.0
                    grp.create_dataset(name=weight_name, data=val[index])
            else:
                grp.create_dataset(name=weight_name, data=weight_values.value)

            #print(weight_name)
            #print(grp[weight_name])

    #h5f.visit(printname)
    h5f.close()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to a .npy file containing a dictionary of parameters'
    )
    FLAGS = parser.parse_args()
    main()