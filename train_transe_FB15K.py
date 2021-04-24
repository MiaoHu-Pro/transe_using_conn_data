import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import codecs

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 20000, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe.ckpt')

print("====get_parameters=====")
out = transe.get_parameters("numpy")
entity_embedding_para = out['ent_embeddings.weight']
relation_embedding_para = out['rel_embeddings.weight']
out_file_title = './benchmarks/FB15K/'
with codecs.open(out_file_title + "out_transE_entity_embedding" + str(transe.dim) + ".txt", "w") as f_embedding:
    for i, e in enumerate(entity_embedding_para):
        f_embedding.write(str(i) + "\t")
        f_embedding.write(str(e.tolist()))
        f_embedding.write("\n")
    f_embedding.close()

with codecs.open(out_file_title + "out_transE_relation_embedding" + str(transe.dim) + ".txt", "w") as f_embedding:
    for i, e in enumerate(relation_embedding_para):
        f_embedding.write(str(i) + "\t")
        f_embedding.write(str(e.tolist()))
        f_embedding.write("\n")
    f_embedding.close()




# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)



