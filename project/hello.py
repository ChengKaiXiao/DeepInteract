from project.datasets.DB5.db5_dgl_dataset import DB5DGLDataset
from project.datasets.DB5.db5_dgl_data_module import DB5DGLDataModule
from project.utils.deepinteract_utils import collect_args, process_args, construct_pl_logger
from project.datasets.DIPS.dips_dgl_dataset import DIPSDGLDataset
from torch.utils.data import DataLoader
import dgl
from typing import List

test_data = DB5DGLDataset(raw_dir="project/datasets/DB5/final/raw")
print(type(test_data))
train_batch = test_data[0]

for obj in train_batch:
    print("Object:",obj)
    print(train_batch[obj])

graph1 = train_batch["graph1"]

print(graph1.ndata['f'])
print(graph1.ndata['f'].size())



# #graph1, graph2, examples_list, filepaths = train_batch[0], train_batch[1], train_batch[2], train_batch[3]

print("type",type(graph1))
# print(graph1)
def dgl_picp_collate(complex_dicts: List[dict]):
    """Assemble a protein complex dictionary batch into two large batched DGLGraphs and a batched labels tensor."""
    batched_graph1 = dgl.batch([complex_dict['graph1'] for complex_dict in complex_dicts])
    batched_graph2 = dgl.batch([complex_dict['graph2'] for complex_dict in complex_dicts])
    examples_list = [complex_dict['examples'] for complex_dict in complex_dicts]
    complex_filepaths = [complex_dict['filepath'] for complex_dict in complex_dicts]
    return batched_graph1, batched_graph2, examples_list, complex_filepaths

test = DIPSDGLDataset(mode='train', raw_dir="project/datasets/DIPS/final/raw")

data_loader = DataLoader(test, batch_size=1, collate_fn=dgl_picp_collate)
train_batch = next(iter(data_loader))
graph1, graph2, examples_list, filepaths = train_batch[0], train_batch[1], train_batch[2], train_batch[3]