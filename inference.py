#code 1:

import torch.nn as nn
class ANNc(nn.Module):
    def __init__(self, emb_size=1024, **kwargs):
        #384
        '''CNN based analogy classifier model.

        It generates a value between 0 and 1 (0 for invalid, 1 for valid) based on four input vectors.
        1st layer (convolutional): 128 filters (= kernels) of size h × w = 1 × 2 with strides (1, 2) and relu activation.
        2nd layer (convolutional): 64 filters of size (2, 2) with strides (2, 2) and relu activation.
        3rd layer (dense, equivalent to linear for PyTorch): one output and sigmoid activation.

        Argument:
        emb_size -- the size of the input vectors'''
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = nn.Conv2d(1, 128, (1,2), stride=(1,2))
        self.conv2 = nn.Conv2d(128, 64, (2,2), stride=(2,2))
        self.linear = nn.Linear(64*(emb_size//2), 1)

    def flatten(self, t):
        '''Flattens the input tensor.'''
        t = t.reshape(t.size()[0], -1)
        return t

    def forward(self,image, p=0):
        """
        
        Expected input shape:
        - a, b, c, d: [batch_size, emb_size]
        """
        #image = torch.stack([a, b, c, d], dim = 2)

        # apply dropout
        if p>0:
            image=F.dropout(image, p)

        x = self.conv1(image.unsqueeze(-3))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        output = torch.sigmoid(x)
        return output

#code 2:
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device)


#code 3
import torch
def load_model(model,file_name='ANNc.pth.tar'):
    #loads model and optimizer 
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["state_dict"])

#code 4
cnn = ANNc().to(device)
load_model(cnn)

#code 5
import re
def clean_value(value):
    # Replace underscores with spaces for each variable
    value = value.replace('_', ' ')
    # Removes the wikipedia article ID from each variable
    value = re.sub(r' Q\d+$', '', value)
    return value

#code 6
def get_embedding(sentences):
    #gets the embeddings, also applies mean pooling, truncation and padding.
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input.attention_mask
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    vector_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return vector_embeddings

#code 7
def inference(x):
    """
    performs a prediction.
    x is a dict that has as a key the 4 identities.(Entity1,Entity2,Entity3,Entity4)
    """
    x = get_embedding(list(x.keys()))
    x_tensor = x.unsqueeze(0).permute(0,2,1)
    y_pred = cnn(x_tensor)
    return 'True' if y_pred >= .5 else 'False'

#code 8
x = {
    'Entity1':'Benoît Biteau',
    'Entity2':'farm operator',
    'Entity3':'Clarence Birdseye',
    'Entity4':'Biologist'
}
inference(x)
# output:'True'
