from tokenizers import BertWordPieceTokenizer
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow_hub as hub

max_seq_length = 384

tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)

class Sample:
    def __init__(self, question, context, start_char_idx=None, answer_text=None, all_answers=None):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self):
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)
        if self.answer_text is not None:
            answer = " ".join(str(self.answer_text).split())
            end_char_idx = self.start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return
            is_char_in_ans = [0] * len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)
            if len(ans_token_idx) == 0:
                self.skip = True
                return
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets

def create_squad_examples(raw_data):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                if "answers" in qa:
                    answer_text = qa["answers"][0]["text"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    squad_eg = Sample(question, context, start_char_idx, answer_text, all_answers)
                else:
                    squad_eg = Sample(question, context)
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples

def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"],
         dataset_dict["input_mask"],
         dataset_dict["input_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y

def predict(context, question):
    data = {"data":
    [
        {"title": "Prediction",
         "paragraphs": [
             {
                 "context": context,
                 "qas": [
                     {"question": question,
                      "id": "Q1"
                      }]
             }
             ]
        }
    ]}
    test_samples = create_squad_examples(data)
    x_test, _ = create_inputs_targets(test_samples)
    pred_start, pred_end = model.predict(x_test)
    for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
        test_sample = test_samples[idx]
        offsets = test_sample.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        pred_ans = None
        if start >= len(offsets):
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
        else:
            pred_ans = test_sample.context[pred_char_start:]
        print("Q: " + test_sample.question)
        print("A: " + pred_ans if pred_ans is not None else 'Cannot be determined from given context')