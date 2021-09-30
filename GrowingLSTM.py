import torch
torch.manual_seed(1)
from helper_functions import get_data, get_dicts, prepare_sequence




class GrowingLSTM(torch.nn.Module):
    def __init__(self, word_to_ix, tag_to_ix):
        super(GrowingLSTM, self).__init__()

        self.new_neuron = -1  # while new

        self.input_size = 3
        self.hidden_size = 1
        self.vocab_size = len(word_to_ix)
        self.tagset_size = len(tag_to_ix)
        self.word_to_ix = word_to_ix
        self.ix_to_word = {value: label for label, value in word_to_ix.items()}
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {value: label for label, value in tag_to_ix.items()}

        self.word_embeddings = torch.nn.Embedding(self.vocab_size, self.input_size)

        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,  # does not apply to hidden or cell states
            dropout=0,
            bidirectional=False,
            proj_size=0,
        )

        self.hidden2tag = torch.nn.Linear(self.hidden_size, self.tagset_size)

    def add_tag(self, tag):
        ix = len(self.tag_to_ix)
        self.tag_to_ix[tag] = ix
        self.ix_to_tag[ix] = tag
        self.tagset_size += 1
        # TODO: create linear layer with old weights + new random weights that lead to new neuron in linear layer
        # TODO: should this be topic of the thesis as this is a totally seperate topic to add new labels to a task?
        # TODO: the topic here should concentrate on increasing the model to learn newly emerging pattern for the current tag set.
        # self.hidden2tag = torch.nn.Linear(self.hidden_dim, self.tagset_size)

    def add_neuron(self):
        print(f"    - The {self.hidden_size + 1}. hidden LSTM neuron has been inserted!")
        self.hidden_size += 1
        self.new_neuron = self.hidden_size - 1  # for learning rate, to learn more on new neuron

        with torch.no_grad():
            self.lstm.hidden_size = self.hidden_size
            # new LSTM weights and biases are initialized as all the weights in pytorch LSTM by default
            # (see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
            # uniform(-sqrt(k), sqrt(k)), k=1/hidden_size
            sqrt_k = (1/self.hidden_size)**0.5

            # "lstm.weight_ih_l0"
            weight_ih_new = torch.FloatTensor(4, self.input_size).uniform_(-sqrt_k, sqrt_k)
            self.lstm.weight_ih_l0 = torch.nn.Parameter(torch.cat((self.lstm.weight_ih_l0.data, weight_ih_new), 0))

            # "lstm.bias_ih_l0"
            bias_ih_new = torch.FloatTensor(4,).uniform_(-sqrt_k, sqrt_k)
            self.lstm.bias_ih_l0 = torch.nn.Parameter(torch.cat((self.lstm.bias_ih_l0.data, bias_ih_new), 0))

            # "lstm.weight_hh_l0"
            weight_hh_new_1 = torch.FloatTensor(4 * (self.hidden_size - 1), 1).uniform_(-sqrt_k, sqrt_k)
            weight_hh_new_2 = torch.FloatTensor(4, self.hidden_size).uniform_(-sqrt_k, sqrt_k)
            tmp = torch.cat((self.lstm.weight_hh_l0.data, weight_hh_new_1), 1)
            self.lstm.weight_hh_l0 = torch.nn.Parameter(torch.cat((tmp, weight_hh_new_2), 0))

            # "lstm.bias_hh_l0"
            bias_hh_new = torch.FloatTensor(4,).uniform_(-sqrt_k, sqrt_k)
            self.lstm.bias_hh_l0 = torch.nn.Parameter(torch.cat((self.lstm.bias_hh_l0.data, bias_hh_new), 0))

            self.hidden2tag.in_features = self.hidden_size
            # new Linear Layer weights are initialized by pytorch Linear as default:
            # (see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
            # uniform(-sqrt(k), sqrt(k)), k=1/input_size
            sqrt_k = (1 / self.input_size) ** 0.5

            # "hidden2tag.weight"
            weight_new = torch.FloatTensor(self.tagset_size, 1).uniform_(-sqrt_k, sqrt_k)
            self.hidden2tag.weight = torch.nn.Parameter(torch.cat((self.hidden2tag.weight.data, weight_new), 1))

            # "hidden2tag.bias"
            # NOTHING TO CHANGE HERE AS IT IS NOT CONNECTED TO OUTPUTS OF LSTM

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.functional.F.log_softmax(tag_space, dim=1)
        return tag_scores

    def predict(self, sentence):
        tag_scores = self.forward(sentence)
        tag_scores_argmax = torch.argmax(tag_scores, dim=1)
        return tag_scores_argmax

    def get_token_level_scores(self, data):
        all_predictions = [self.predict(prepare_sequence(sent, self.word_to_ix)) for (sent, _) in data]
        all_gold_targets = [prepare_sequence(gold_targets, self.tag_to_ix) for (_, gold_targets) in data]
        fp = {}
        fn = {}
        tp = {}
        tn = {}
        for tag in self.tag_to_ix:
            fp[tag] = 0
            fn[tag] = 0
            tp[tag] = 0
            tn[tag] = 0
        for (prediction, gold_targets) in zip(all_predictions, all_gold_targets):
            for (word_prediction, word_gold_target) in zip(prediction, gold_targets):
                word_prediction = word_prediction.item()
                word_gold_target = word_gold_target.item()

                for tag in self.tag_to_ix:
                    tag_ix = self.tag_to_ix[tag]
                    if word_prediction == tag_ix and word_gold_target == tag_ix:  # tp
                        tp[tag] += 1
                    elif word_prediction != tag_ix and word_gold_target == tag_ix:  # fn
                        fn[tag] += 1
                    elif word_prediction == tag_ix and word_gold_target != tag_ix:  # fp
                        fp[tag] += 1
                    else:  # tn
                        tn[tag] += 1
        accuracies = {}
        precisions = {}
        recalls = {}
        f1_score = {}
        for tag in self.tag_to_ix:
            accuracies[tag] = (tp[tag] + tn[tag])/(tp[tag]+fp[tag]+fn[tag]+tn[tag]) if (tp[tag]+fp[tag]+fn[tag]+tn[tag]) > 0 else 0
            precisions[tag] = tp[tag]/(tp[tag]+fp[tag]) if (tp[tag]+fp[tag]) > 0 else 0
            recalls[tag] = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0
            f1_score[tag] = 2 * (precisions[tag] * recalls[tag]) / (precisions[tag] + recalls[tag]) if (precisions[tag] + recalls[tag]) > 0 else 0
        return accuracies, precisions, recalls, f1_score

    def print_token_level_scores(self, data, only_f1_score=True):
        accuracies, precisions, recalls, f1_score = self.get_token_level_scores(data)
        print("##############################################################################")
        if not only_f1_score:
            print(f"Accuracies: {accuracies}")
            print(f"Precisions: {precisions}")
            print(f"Recalls: {recalls}")
        print(f"F1-Score: {f1_score}")
        print("##############################################################################")


data_train = get_data(100)
word_to_ix, tag_to_ix = get_dicts(data_train)
model = GrowingLSTM(word_to_ix, tag_to_ix)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model.add_neuron()

for epoch in range(1000):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in data_train:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()

        #optimizer.step()
        with torch.no_grad():
            for p in model.named_parameters():
                (name, param) = p
                if name == "lstm.weight_ih_l0":
                    new_val = param - 0.1 * param.grad
                elif name == "lstm.weight_hh_l0":
                    new_val = param - 0.1 * param.grad
                elif name == "lstm.bias_ih_l0":
                    new_val = param - 0.1 * param.grad
                elif name == "lstm.bias_hh_l0":
                    new_val = param - 0.1 * param.grad
                elif name == "hidden2tag.weight":
                    new_val = param - 0.1 * param.grad
                elif name == "hidden2tag.bias":
                    new_val = param - 0.1 * param.grad
                else:
                    new_val = param - 0.1 * param.grad
                param.copy_(new_val)
                #param.grad.zero_()  # dont necessary because of model.zero_grad()

    if (epoch % 50) == 0:
        print("epoch " + str(epoch))
        with torch.no_grad():
            model.print_token_level_scores(data_train)

    if (epoch % 50) == 0 and epoch > 0:
        print("epoch " + str(epoch))
        with torch.no_grad():
            model.add_neuron()
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(data_train[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    #print(tag_scores)
    model.print_token_level_scores(data_train)