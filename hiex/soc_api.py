from .soc_algo import _SamplingAndOcclusionAlgo
from .lm import BiGRULanguageModel
from .train_lm import do_train_lm
import os, logging, torch, pickle
import json
from utils.misc import get_possible_nw_tokens
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class SamplingAndOcclusionExplain:
    def __init__(self, model, configs, tokenizer, output_path, device, lm_dir=None, train_dataloader=None,
                 dev_dataloader=None, vocab=None, model_type='roberta'):
        self.configs = configs
        self.model = model
        self.lm_dir = lm_dir
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.vocab = vocab
        self.output_path = output_path
        self.device = device
        self.hiex = configs.hiex
        self.tokenizer = tokenizer

        self.pad_token = tokenizer.pad_token_id

        self.lm_model = self.detect_and_load_lm_model(self.pad_token)

        self.algo = _SamplingAndOcclusionAlgo(model, tokenizer, self.lm_model, output_path, configs)

        self.use_padding_variant = configs.use_padding_variant
        try:
            self.output_file = open(self.output_path, 'w' if not configs.hiex else 'wb')
        except FileNotFoundError:
            self.output_file = None
        self.output_buffer = []

        # for explanation regularization
        self.neutral_words_file = configs.neutral_words_file
        self.neutral_words_ids = None
        self.neutral_words = None

        self.batched = configs.expl_batched
        self.expl_batch_size = configs.expl_batch_size
        #self.debug = debug

    def detect_and_load_lm_model(self, pad_token):
        if not self.lm_dir:
            self.lm_dir = 'runs/lm/'
        if not os.path.isdir(self.lm_dir):
            os.mkdir(self.lm_dir)

        file_name = None
        for x in os.listdir(self.lm_dir):
            if x.startswith('best'):
                file_name = x
                break
        if not file_name:
            self.train_lm(pad_token)
            for x in os.listdir(self.lm_dir):
                if x.startswith('best'):
                    file_name = x
                    break
        lm_model = torch.load(open(os.path.join(self.lm_dir,file_name), 'rb'))
        return lm_model

    def train_lm(self, pad_token):
        logger.info('Missing pretrained LM. Now training')
        model = BiGRULanguageModel(self.configs, vocab=self.vocab, device=self.device,
                                   pad_token=pad_token).to(self.device)
        do_train_lm(model, lm_dir=self.lm_dir, lm_epochs=20,
                    train_iter=self.train_dataloader, dev_iter=self.dev_dataloader)

    def word_level_explanation_bert(self, input_ids, input_mask, token_type_ids, label=None):
        # requires batch size is 1
        # get sequence length
        i = 0
        while i < input_ids.size(1) and input_ids[0,i] != 0: # pad
            i += 1
        inp_length = i
        # do not explain [CLS] and [SEP]
        spans, scores = [], []
        for i in range(1, inp_length-1, 1):
            span = (i, i)
            spans.append(span)
            if not self.use_padding_variant:
                score = self.algo.do_attribution(input_ids, input_mask, token_type_ids, span, label)
            else:
                score = self.algo.do_attribution_pad_variant(input_ids, input_mask, token_type_ids, span, label)
            scores.append(score)
        inp = input_ids.view(-1).cpu().numpy()
        s = self.algo.repr_result_region(inp, spans, scores)
        self.output_file.write(s + '\n')

    def hierarchical_explanation_bert(self, input_ids, input_mask, token_type_ids, label=None):
        tab_info = self.algo.do_hierarchical_explanation(input_ids, input_mask, token_type_ids, label)
        self.output_buffer.append(tab_info)
        # currently store a pkl after explaining each instance
        self.output_file = open(self.output_path, 'w' if not self.hiex else 'wb')
        pickle.dump(self.output_buffer, self.output_file)
        self.output_file.close()

    def _initialize_neutral_words(self):
        neutral_words, neutral_words_ids = get_possible_nw_tokens(self.neutral_words_file, self.tokenizer)
        logger.info(neutral_words)
        self.neutral_words = neutral_words
        self.neutral_words_ids = neutral_words_ids
        assert neutral_words

    def find_neutral_word_positions(self, input_np):
        positions = []
        candidate_nw = []
        input_ids = set(input_np.tolist())
        for token_ids in self.neutral_words_ids:
            if input_ids.issuperset(token_ids):
                candidate_nw.append(token_ids)

        for start in range(len(input_np)):
            if input_np[start] == self.tokenizer.pad_token_id:
                break
            for token_ids in candidate_nw:
                if start + len(token_ids) > len(input_np):
                    continue
                matched = True
                for j in range(len(token_ids)):
                    if input_np[start + j] != token_ids[j]:
                        matched = False
                        break
                if matched:
                    positions.append((start, start + len(token_ids) - 1))
        return list(set(positions))

    def compute_explanation_loss(self, *args, **kwargs):
        if self.neutral_words is None:
            self._initialize_neutral_words()
        if self.batched:
            return self._compute_explanation_loss_batched(*args, **kwargs)
        else:
            return self._compute_explanation_loss_unbatched(*args, **kwargs)

    def _compute_explanation_loss_batched(self, input_ids_batch, input_mask_batch, token_type_ids_batch, label_ids_batch,
                                         do_backprop=False, skip_i=None, call_fn=None, reg_strength=None, **kwargs):
        batch_size = input_ids_batch.size(0)
        input_enb, input_ex = [], []
        input_mask_enb, input_mask_ex = [], []
        token_type_ids_enb = []
        for b in range(batch_size):
            if skip_i and b in skip_i:
                continue
            input_ids, input_mask, token_type_ids, label_ids = input_ids_batch[b], \
                                                               input_mask_batch[b], \
                                                               token_type_ids_batch[b] if token_type_ids_batch is not None else None, \
                                                               label_ids_batch[b]
            mask_regions = self.find_neutral_word_positions(input_ids.cpu().numpy())
            for x_region in mask_regions:
                input_enb.append(input_ids)
                input_mask_enb.append(input_mask)
                token_type_ids_enb.append(token_type_ids)

                input_ids_, input_mask_, token_type_ids_ = input_ids.clone(), input_mask.clone(), token_type_ids.clone() if token_type_ids is not None else None
                input_ids_[x_region[0]: x_region[0] + 1] = self.tokenizer.pad_token_id
                input_mask_[x_region[0]: x_region[1] + 1] = 0

                input_ex.append(input_ids_)
                input_mask_ex.append(input_mask_)

        if input_enb:
            # slice the batch to prevent oom
            all_scores = []
            for start in range(0, len(input_enb), self.expl_batch_size):
                stop = start + self.expl_batch_size
                if stop > len(input_enb):
                    stop = len(input_enb)

                input_enb_batch, input_mask_enb_batch, token_type_ids_batch = torch.stack(input_enb[start:stop],0).detach(), torch.stack(input_mask_enb[start:stop],0).detach(), \
                                                                            torch.stack(token_type_ids_enb[start:stop],0).detach() if token_type_ids_enb[0] is not None else None
                input_ex_batch, input_mask_ex_batch = torch.stack(input_ex[start:stop], 0).detach(), torch.stack(input_mask_ex[start:stop],0).detach()

                logits_enb = self.model(
                    input_ids=input_enb_batch,
                    token_type_ids=token_type_ids_batch[:, :input_enb_batch.size(1)] if token_type_ids_batch is not None else None,
                    attention_mask=input_mask_enb_batch,
                )
                logits_ex = self.model(
                    input_ids=input_ex_batch,
                    token_type_ids=token_type_ids_batch[:, :input_enb_batch.size(1)] if token_type_ids_batch is not None else None,
                    attention_mask=input_mask_ex_batch
                )
                if type(logits_enb) is tuple:
                    logits_enb = logits_enb[0]
                    logits_ex = logits_ex[0]

                if self.configs.softmax_contrib:
                    logits_enb = F.softmax(logits_enb, -1)
                    logits_ex = F.softmax(logits_ex, -1)
                    contrib_logits = logits_enb - logits_ex
                else:
                    # TODO: for debug here
                    contrib_logits = logits_enb - logits_ex

                if contrib_logits.size(1) == 2:  # binary
                    contrib_score = contrib_logits[:, 1] - contrib_logits[:, 0]
                else:
                    contrib_score_sec_max, _ = contrib_logits[:, 1:].max(-1)
                    contrib_score = contrib_score_sec_max - contrib_logits[:, 0]

                score = contrib_score.sum()

                if reg_strength is None:
                    reg_strength = self.configs.reg_strength

                if self.configs.reg_abs:
                    score = reg_strength * abs(score)
                else:
                    score = reg_strength * (score ** 2)

                if do_backprop:
                    score.backward()
                all_scores.extend(contrib_score.detach().cpu().numpy().tolist())

            return sum(all_scores), len(all_scores)
        else:
            return 0., 0

    def _compute_explanation_loss_unbatched(self, input_ids_batch, input_mask_batch, token_type_ids_batch, label_ids_batch,
                                 do_backprop=False, skip_i=None, call_fn=None, reg_strength=None,
                                 importance_target='output'):
        batch_size = input_ids_batch.size(0)
        neutral_word_scores, cnt = [], 0
        for b in range(batch_size):
            if skip_i and b in skip_i:
                continue
            input_ids, input_mask, token_type_ids, label_ids = input_ids_batch[b], \
                                                            input_mask_batch[b], \
                                                            token_type_ids_batch[b] if token_type_ids_batch is not None else None, \
                                                            label_ids_batch[b]
            #nw_positions = []
            #for i in range(len(input_ids)):
            #    word_id = input_ids[i].item()
            #    if word_id in self.neutral_words_ids:
            #        nw_positions.append(i)
            # only generate explanations for neutral words

            mask_regions = self.find_neutral_word_positions(input_ids.cpu().numpy())

            for x_region in mask_regions:
                #word_id = input_ids[i].item()
                #if word_id in self.neutral_words_ids:

                #score = self.algo.occlude_input_with_masks_and_run(input_ids, input_mask, token_type_ids,
                #                                                   [x_region], nb_region, label_ids,
                #                                                    return_variable=True)
                if not self.configs.use_padding_variant:
                    score = self.algo.do_attribution(input_ids, input_mask, token_type_ids, x_region, label_ids,
                                                     return_variable=True, importance_target=importance_target)
                else:
                    score = self.algo.do_attribution_pad_variant(input_ids, input_mask, token_type_ids,
                                                                 x_region, label_ids, return_variable=True,
                                                                 importance_target=importance_target
                                                                )

                if reg_strength is None:
                    reg_strength = self.configs.reg_strength

                if self.configs.reg_abs:
                    score = reg_strength * abs(score)
                else:
                    score = reg_strength * (score ** 2)

                if do_backprop:
                    score.backward()
                if call_fn is not None:
                    call_fn()

                neutral_word_scores.append(score.item())

        if neutral_word_scores:
            return sum(neutral_word_scores), len(neutral_word_scores)
        else:
            return 0., 0
