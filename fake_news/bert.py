# #導入BERT環境部署
# pip install transformers tqdm boto3 requests regex -q
import os
import pandas as pd
import torch 
from transformers import BertTokenizer 
from IPython.display import clear_output
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

priject_dir = os.path.dirname(__file__)  # get current directory
module_dir = os.path.join(priject_dir, 'model')

def getBERTPredict(ans):
    return_data={}
    #網頁輸入資料，看這部分就要麻煩看如何改成Django的網頁輸入值
    input_text=ans
    # print(f'input_text:{input_text}')
    input_id='1'
    module_bert_dir = os.path.join(module_dir, 'bert')

    #讀取已建好CSV檔
    df_input = pd.read_csv(os.path.join(module_bert_dir, "input_project_BERT.csv"))
    df_input=df_input.append({'text' : input_text ,'Id' : input_id} , ignore_index=True)
    SAMPLE_FRAC = 1
    df_input = df_input.sample(frac=SAMPLE_FRAC, random_state=9527)
    df_input.columns = ["text",'Id']
    df_input.to_csv(os.path.join(module_bert_dir, "input_project.tsv"), sep="\t", index=False)
    # print("輸入樣本數：", len(df_input))

    #指定繁簡中⽂ BERT-BASE 預訓練模型
    PRETRAINED_MODEL_NAME = "bert-base-chinese" 
    # 取得此預訓練模型所使⽤的tokenizer 
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    clear_output() 
    print("PyTorch版本： " , torch.__version__)

    
    class FakeNewsDataset(Dataset):
        # 讀取前處理後的 tsv 檔並初始化一些參數
        def __init__(self, mode, tokenizer):
            assert mode in ["train_project", "test_project","input_project"]  # 一般訓練你會需要 dev set
            self.mode = mode
            # 大數據你會需要用 iterator=True
            # print(f'mode={mode}')
            self.df = pd.read_csv(os.path.join(module_bert_dir, mode + ".tsv"), sep="\t").fillna("")
            self.len = len(self.df)
            self.label_map = { 'fn' : 0, 'tn' : 1}
            self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
            # print(self.label_map[label])
            # print(type(self.df.iloc[2, :1].str))
            # print(self.df.iloc[10, :].values)
        # 定義回傳一筆訓練 / 測試數據的函式
        def __getitem__(self, idx):
            if self.mode == "test_project" :
                text_a = self.df.iloc[idx, :1].values[0]
                label_tensor = None
            if self.mode == "input_project":
                text_a = self.df.iloc[idx, :1].values[0]
                label_tensor = None
            else:
                text_a,label = self.df.iloc[idx, :].values
                # 將 label 文字也轉換成索引方便轉換成 tensor
                label_id = self.label_map[label]
                label_tensor = torch.tensor(label_id)
                
            # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
            word_pieces = ["[CLS]"]
            tokens_a = self.tokenizer.tokenize(text_a)
            word_pieces += tokens_a 
            len_a = len(word_pieces)
            
            
            # 將整個 token 序列轉換成索引序列
            ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
            tokens_tensor = torch.tensor(ids)
            
            # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
            segments_tensor = torch.tensor([0] * len_a , 
                                            dtype=torch.long)
            
            return (tokens_tensor, segments_tensor, label_tensor)
        
        
        def __len__(self):
            return self.len


    # 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
    # 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
    # - tokens_tensor
    # - segments_tensor
    # - label_tensor
    # 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
    def create_mini_batch(samples):
        tokens_tensors = [s[0] for s in samples]
        segments_tensors = [s[1] for s in samples]
        
        # 測試集有 labels
        if samples[0][2] is not None:
            label_ids = torch.stack([s[2] for s in samples])
        else:
            label_ids = None
        
        # zero pad 到同一序列長度
        tokens_tensors = pad_sequence(tokens_tensors, 
                                    batch_first=True)
        segments_tensors = pad_sequence(segments_tensors, 
                                        batch_first=True)
        
        # attention masks，將 tokens_tensors 裡頭不為 zero padding
        # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, 
                                    dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(
            tokens_tensors != 0, 1)
        
        return tokens_tensors, segments_tensors, masks_tensors, label_ids

    model=torch.load(os.path.join(module_bert_dir, 'BERT_LEE.pth'),map_location='cpu')

    def get_predictions(model, dataloader, compute_acc=False):
        predictions = None
        correct = 0
        total = 0
        
        with torch.no_grad():
            # 遍巡整個資料集
            for data in dataloader:
                # 將所有 tensors 移到 GPU 上
                if next(model.parameters()).is_cuda:
                    data = [t.to("cuda:0") for t in data if t is not None]
                
                
                # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
                # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
                tokens_tensors, segments_tensors, masks_tensors = data[:3]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors)
                
                logits = outputs[0]
                _, pred = torch.max(logits.data, 1)
                
                # 用來計算訓練集的分類準確率
                if compute_acc:
                    labels = data[3]
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    
                # 將當前 batch 記錄下來
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))
        
        if compute_acc:
            acc = correct / total
            return predictions, acc
        return predictions


    inputset = FakeNewsDataset("input_project", tokenizer=tokenizer)
    inputloader = DataLoader(inputset, batch_size=1, 
                            collate_fn=create_mini_batch)

    predictions = get_predictions(model, inputloader)
    index_map = {v: k for k, v in inputset.label_map.items()}
    # df_test_og = pd.read_csv("input_project_BERT.csv")
    df = pd.DataFrame({"Category": predictions.tolist()})
    df['Category'] = df.Category.apply(lambda x: index_map[x])
    df_pred = pd.concat([inputset.df.loc[:, ["Id"]], 
                            df.loc[:, 'Category']], axis=1)
    # data['success'] = True 
    # print(f'df_pred===={df_pred}')
    if df_pred.iloc[0:1]['Category'].values=='tn':
        return_data['result'] = True
    else:
        return_data['result'] = False
    return_data['confidence'] = 0.0
    return_data['success'] = True 
    return return_data
