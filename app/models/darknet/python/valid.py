from voc_eval import voc_eval
import numpy as np
import matplotlib.pyplot as plt


print(voc_eval('/home/qingpeng/git/darknet/results/comp4_det_test_{}.txt', '/hdd2/qingpeng/object_detection/xmls/{}.xml', '/home/qingpeng/git/darknet/data/food/test.txt', 'food', '.'))


# import os
# map_ = 0
# # classnames填写训练模型时定义的类别名称
# classnames = ['food']
# for classname in classnames:
#     rec, prec, ap  = voc_eval('/home/qingpeng/git/darknet/results/comp4_det_test_{}.txt', '/hdd2/qingpeng/object_detection/xmls/{}.xml', '/home/qingpeng/git/darknet/data/food/test.txt', classname, '.')
#     map_ += ap
#     print ('%s' % (classname + '_ap:')+'%s' % ap)
# # 删除临时的dump文件
# # if(os.path.exists("annots.pkl")):
# #     os.remove("annots.pkl")
# #     print("cache file:annots.pkl has been removed!")
# # 打印map
# mAP = map_/len(classnames)
# print ('mAP:%s' % mAP)

    
rec, prec, ap  = voc_eval('/home/qingpeng/git/darknet/results/comp4_det_test_{}.txt', '/hdd2/qingpeng/object_detection/xmls/{}.xml', '/home/qingpeng/git/darknet/data/food/test.txt', 'food', '.')
print("[*]food  mAP: %s rec: %s prec: %s" % (ap,rec,prec))

plt.title('Result Analysis')
plt.plot(prec, rec, color='blue', label='food')
plt.legend() # 显示图例
plt.xlabel('recall')
plt.ylabel('precion')
plt.savefig('./result.png',transparent=True,pad_inches=0,dpi=300,bbox_inches='tight')
# plt.show()
