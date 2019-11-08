# Load the model in fairseq
import torch
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained(model_name_or_path='./roberta.base', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
tokens = roberta.encode('Hello world!')
print(tokens)

# import torch
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

# roberta.eval()  # disable dropout (or leave in train mode to finetune)
# tokens = roberta.encode('Hello world!')
# print(tokens)
# assert tokens.tolist() == [0, 31414, 232, 328, 2]
# roberta.decode(tokens)  # 'Hello world!'