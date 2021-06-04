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

    test_cases = ["<s> ĠMuslims </s>",
         '<s> ĠThank Ġyou ! ĠFighting Ġa Ġdecade Ġlong Ġlosing Ġbattle Ġagainst Ġhor rid ly Ġbad Ġcholesterol . </s>',
         '<s> ĠApparently , Ġthe ĠEU Ġis Ġthreatening ĠCentral ĠEuropean Ġstates Ġwith Ġuse Ġof Ġforce . </s>',
         '<s> ĠThat Ġwas Ġa Ġsupplemental Ġpsy - op Ġto Ġmove Ġthe Ġnews Ġnarrative Ġaway Ġfrom Ġthe ĠPark land Ġstate - sponsored Ġgoat Ġrode o . </s>',
         '<s> ĠI Ġusually Ġfeel Ġthe Ġneed Ġto Ġdown vote Ġwhen ĠI Ġsee Ġentire Ġcomment Ġthreads Ġcalling Ġusers Ġnig gers Ġand Ġk ikes Ġwith Ġ80 Ġup votes Ġand Ġzero Ġdown votes . </s>',
         "<s> ĠSo Ġlet Ġme Ġget Ġthis Ġstraight Ġ ĠThey 're Ġgonna Ġput Ġyour Ġkids Ġin Ġprison Ġwith Ġrapists Ġand Ġkillers Ġto Ġprotect Ġthem Ġfrom Ġinternet Ġpredators ! ?! Ġ ĠBut Ġthe ĠClintons , ĠPod est as , Ġand Ġwhoever 's Ġin ĠNX IV M Ġremain Ġfree ! ?! Ġ ĠWhen Ġare Ġwe Ġgonna Ġchange Ġthese Ġlaws Ġand Ġget Ġour Ġpriorities Ġstraight ? </s>",
         "<s> ĠOK Âł ĠIF Ġyou Ġwant Ġto Ġgo Ġdeeper Ġdown Ġthe ĠRabbit ĠHole Ġre - ĠChristopher ĠStory , Ġthis Ġwill Ġblow Ġyour Ġmind . Âł ĠNotice ĠLord ĠSass oon , Ġit Ġwas Ġhis Ġfamily Ġin ĠHong ĠKong Ġwho Ġworked Ġwith ĠDavid ĠCameron 's ĠGreat ĠGreat ĠGrand ĠFather ĠE win ĠCameron Ġin Ġthe Ġopium Ġtrade Ġwith ĠChina .... HS BC ÂłÂł Ġit Ġhas Ġnever Ġchanged . Âł Ġfour Ġparts ÂłÂł Ġmind - bl owing Ġ ĠTHE ĠI LL UM IN AT IS ĠSEC RET ĠPLAN ĠTO ĠBRE AK DOWN ĠWORLD ĠE CON OM Y ĠBY ĠIN JECT ING ĠON ĠTO ĠTHE ĠPRE F AB RIC ATED ĠGL OB AL ĠCR ISIS ĠTHE ĠSEC OND ĠWORLD ĠWAR ĠNA ZI ĠAND ĠJ AP AN ESE ĠLO OT ED ĠGOLD ĠACT </s>",
         '<s> ĠJust Ġlike Ġthe Ġcompassion ĠHe Ġshowed Ġin Ġthe ĠCamp , ĠHuh ĠAss hole Ġ!! </s>',
         '<s> ĠPan icked ĠWeiner ĠD umped ĠHis ĠL aptop Ġ& ĠPh ones Ġto ĠPrivate ĠInvestigator Ġbefore ĠFBI ĠRaid ; ĠWho ĠElse ĠHas ĠThe ĠHillary ĠFiles ? </s>',
         '<s> ĠIm Ġaware Ġof ĠTommy Ġa ĠBritish Ġpatriot . Ġ ĠHe Ġinterviewed Ġyour Ġonly Âł Ġnon Ġcorrupt Ġlord Ġ ĠYOU ĠNEED ĠT ĠSUPPORT ĠHIM Ġ ĠMAR CH ĠIN ĠTHE ĠST RE ETS Ġ ĠSP E AK ĠIN ĠTHE ĠPU BS Ġ ĠR ALLY ĠYOUR ĠTR O OPS Ġ ĠZ ERO ĠDAY ĠIS ĠCOM ING ĠF AST . </s>']

    res = []
    for i, test_case in enumerate(test_cases):
        nw_positions = find_neutral_word_positions_for_metrics(terms, test_case)
        res.append(nw_positions)
    print(res)