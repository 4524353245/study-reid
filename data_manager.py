from __future__ import print_function, absolute_import

import os
import os.path as osp
import numpy as np
import glob
import re

# from IPython import embed
# embed()

# 以读取 market1501 数据集为例
class Market1501(object):
    
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)

    """
    dataset_dir = 'market1501'

    def __init__(self,root='data',**kwargs):
        self.dataset_dir = osp.join(root,self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir,'query')
        self.gallery_dir = osp.join(self.dataset_dir,'bounding_box_test')

        self._check_before_run()
        # 获取训练集的数据
        train,num_train_pids,num_train_imgs=self._process_dir(self.train_dir,relabel=True) # 对训练集数据进行处理
        # 查询集
        query,num_query_pids,num_query_imgs=self._process_dir(self.query_dir,relabel=False)
        # 被查集
        gallery,num_gallery_pids,num_gallery_imgs=self._process_dir(self.gallery_dir,relabel=False)
        
        num_total_pids = num_train_imgs + num_query_imgs # 所有的ID是训练集加上测试集
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery
 
        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        #print(num_train_pids,num_train_images)
    def _check_before_run(self):   # 在运行前检查路径是否存在
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("{} is not available".format(self.dataset_dir))

    # 文件路径 标注信息 ID、CAMID、图片数量
    # 训练集是要 relabel 的 
    def _process_dir(self,dir_path,relabel = False): # 给ID重新排序默认关闭
        img_paths = glob.glob(osp.join(dir_path,'*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set() # set类型不重复
        for  img_path in img_paths:
            pid,_ = map(int,pattern.search(img_path).groups())
            if pid == -1: continue # Marked1501有一些垃圾数据，是单纯的背景，id为-1。若遇到直接跳过
            pid_container.add(pid) # 751个人则集合有751个id, id编号区间为0-1501(不是全部)
        # print(pid_container)
        pid2label = {pid:label for label,pid in enumerate(pid_container)} # 集合存入映射-> 在集合中的索引(新id):pid(原id)
        
        # print(pid2label)
        dataset = []
        for  img_path in img_paths:
            pid, camid= map(int,pattern.search(img_path).groups())
            if pid == -1: continue
            assert 0<= pid <=1501
            assert 1<= camid <=6
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path,pid,camid))
        num_pids = len(pid_container)
        num_images = len(img_paths)
        return dataset,num_pids,num_images


if __name__ == '__main__':
    data = Market1501(root='C:\\Users\\surface\\Desktop')
    print("in")
    
   

