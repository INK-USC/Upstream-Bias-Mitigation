import os

class DotDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_possible_nw_tokens(nw_filename, roberta_tokenizer, tokenize=True):
    no_case = False
    if 'NO_CASE' in os.environ and str(os.environ['NO_CASE']) == str(1):
        no_case = True
        print('no case is true')

    f = open(nw_filename)
    neutral_words = []
    neutral_words_ids = []
    for line in f.readlines():
        word = line.strip().split('\t')[0]
        variations = []
        if tokenize:
            canonical, canonical_cap = roberta_tokenizer.tokenize(' ' + word), \
                                       roberta_tokenizer.tokenize(' ' + word.capitalize())
            #canonical2, canonical_cap2 = roberta_tokenizer.tokenize(word), \
            #                             roberta_tokenizer.tokenize(word.capitalize())
            #variations.append(canonical2)
            #variations.append(canonical_cap2)
        else:
            canonical, canonical_cap = [word], [word.capitalize()]
        variations.append(canonical)
        variations.append(canonical_cap)
        # if len(canonical) > 1:
        #    canonical.sort(key=lambda x: -len(x))
        #    print(canonical)
        # word = canonical[0]
        # neutral_words.append(word)
        # neutral_words_ids.add(self.vocab[word])

        if not no_case:
            if tokenize:
                can_pl = roberta_tokenizer.tokenize(' ' + word + 's')
                can_cap_pl = roberta_tokenizer.tokenize(' ' + word.capitalize() + 's')
                #can_pl2 = roberta_tokenizer.tokenize(word + 's')
                #can_cap_pl2 = roberta_tokenizer.tokenize(word.capitalize() + 's')
                #if can_pl2[:len(canonical2)] != canonical2:
                #    variations.append(can_pl2)
                #if can_cap_pl2[:len(canonical_cap2)] != canonical_cap2:
                #    variations.append(can_cap_pl2)
            else:
                can_pl, can_cap_pl = [word + 's'], [word.capitalize() + 's']

            if can_pl[:len(canonical)] != canonical:
                variations.append(can_pl)
            if can_cap_pl[:len(canonical_cap)] != canonical_cap:
                variations.append(can_cap_pl)

        variations = list(set([tuple(_) for _ in variations]))

        neutral_words.extend(variations)
        for w in variations:
            neutral_words_ids.append(tuple([roberta_tokenizer.encoder[_] for _ in w]))
        neutral_words_ids = list(set(neutral_words_ids))
    return neutral_words, neutral_words_ids


def find_neutral_word_positions(input_tokens, neutral_words, tokenizer, return_matched=False):
    positions = []
    candidate_nw = []
    matched_terms = []
    for tokens in neutral_words:
        if set(input_tokens).issuperset(tokens):
            candidate_nw.append(tokens)

    for start in range(len(input_tokens)):
        if input_tokens[start] == tokenizer.pad_token_id or input_tokens[start] == tokenizer.pad_token:
            break
        for token_ids in candidate_nw:
            if start + len(token_ids) > len(input_tokens):
                continue
            matched = True
            for j in range(len(token_ids)):
                if input_tokens[start + j] != token_ids[j]:
                    matched = False
                    break
                j += 1
            if matched:
                positions.append((start, start + len(token_ids) - 1))
                matched_terms.append(token_ids)
    if return_matched:
        return list(set(positions)), matched_terms
    else:
        return list(set(positions))

def load_group_identifiers_for_metrics(filename='datasets/identity.csv', tokenizer=None):
    f = open(filename)
    res = []
    for line in f.readlines():
        word = line.split('\t')[0]
        tokens = tokenizer.tokenize(' ' + word)
        res.append(tokens)
        res.append(tokenizer.tokenize(' ' + word.capitalize()))
        res.append(tokenizer.tokenize(' ' + word + 's'))
        res.append(tokenizer.tokenize(' ' + word.capitalize() + 's'))
    return res

def find_neutral_word_positions_for_metrics(neutral_words, sent):
    sent = sent.split()
    #print(sent)
    positions = []
    for start in range(len(sent)):
        for tokens in neutral_words:
            matched = True
            for j in range(len(tokens)):
                if sent[start + j] != tokens[j]:
                    matched = False
                    break
            if matched:
                positions.append((start, start + len(tokens) - 1))
    return positions

if __name__ == '__main__':
    from transformers.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    terms = load_group_identifiers_for_metrics(tokenizer=tokenizer,filename='datasets/identity_gab.csv')

    test_cases = ["<s> ??Muslims </s>",
         '<s> ??Thank ??you ! ??Fighting ??a ??decade ??long ??losing ??battle ??against ??hor rid ly ??bad ??cholesterol . </s>',
         '<s> ??Apparently , ??the ??EU ??is ??threatening ??Central ??European ??states ??with ??use ??of ??force . </s>',
         '<s> ??That ??was ??a ??supplemental ??psy - op ??to ??move ??the ??news ??narrative ??away ??from ??the ??Park land ??state - sponsored ??goat ??rode o . </s>',
         '<s> ??I ??usually ??feel ??the ??need ??to ??down vote ??when ??I ??see ??entire ??comment ??threads ??calling ??users ??nig gers ??and ??k ikes ??with ??80 ??up votes ??and ??zero ??down votes . </s>',
         "<s> ??So ??let ??me ??get ??this ??straight ?? ??They 're ??gonna ??put ??your ??kids ??in ??prison ??with ??rapists ??and ??killers ??to ??protect ??them ??from ??internet ??predators ! ?! ?? ??But ??the ??Clintons , ??Pod est as , ??and ??whoever 's ??in ??NX IV M ??remain ??free ! ?! ?? ??When ??are ??we ??gonna ??change ??these ??laws ??and ??get ??our ??priorities ??straight ? </s>",
         "<s> ??OK ???? ??IF ??you ??want ??to ??go ??deeper ??down ??the ??Rabbit ??Hole ??re - ??Christopher ??Story , ??this ??will ??blow ??your ??mind . ???? ??Notice ??Lord ??Sass oon , ??it ??was ??his ??family ??in ??Hong ??Kong ??who ??worked ??with ??David ??Cameron 's ??Great ??Great ??Grand ??Father ??E win ??Cameron ??in ??the ??opium ??trade ??with ??China .... HS BC ???????? ??it ??has ??never ??changed . ???? ??four ??parts ???????? ??mind - bl owing ?? ??THE ??I LL UM IN AT IS ??SEC RET ??PLAN ??TO ??BRE AK DOWN ??WORLD ??E CON OM Y ??BY ??IN JECT ING ??ON ??TO ??THE ??PRE F AB RIC ATED ??GL OB AL ??CR ISIS ??THE ??SEC OND ??WORLD ??WAR ??NA ZI ??AND ??J AP AN ESE ??LO OT ED ??GOLD ??ACT </s>",
         '<s> ??Just ??like ??the ??compassion ??He ??showed ??in ??the ??Camp , ??Huh ??Ass hole ??!! </s>',
         '<s> ??Pan icked ??Weiner ??D umped ??His ??L aptop ??& ??Ph ones ??to ??Private ??Investigator ??before ??FBI ??Raid ; ??Who ??Else ??Has ??The ??Hillary ??Files ? </s>',
         '<s> ??Im ??aware ??of ??Tommy ??a ??British ??patriot . ?? ??He ??interviewed ??your ??only ???? ??non ??corrupt ??lord ?? ??YOU ??NEED ??T ??SUPPORT ??HIM ?? ??MAR CH ??IN ??THE ??ST RE ETS ?? ??SP E AK ??IN ??THE ??PU BS ?? ??R ALLY ??YOUR ??TR O OPS ?? ??Z ERO ??DAY ??IS ??COM ING ??F AST . </s>']

    res = []
    for i, test_case in enumerate(test_cases):
        nw_positions = find_neutral_word_positions_for_metrics(terms, test_case)
        res.append(nw_positions)
    print(res)