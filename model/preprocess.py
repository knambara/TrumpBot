import argparse
import pickle
import re

from tqdm import tqdm

sentence_regex = re.compile(r'<(?P<speaker>.*)>\s*<(?P<sentiment>.*)>\s*(?P<sentence>.*)\s*')


def parse_sentence(sentence: str):
    """
    >>> s = "<Sean Hannity> <Positive> All right. Joining us now live on the phone is, from the White House, President Donald Trump. Mr. President, I know you've been busy. I'm sure you never predicted this as part of any presidency. Thank you for spending time with us."
    >>> components = parse_sentence(s)
    >>> components['speaker']
    'Sean Hannity'
    >>> components['sentiment']
    'Positive'

    >>> s = "<Donald Trump> <Positive> I will never let you down. I can say that. Never. Thank you. I am honored to be here with leaders from across the country and all around the world who are all united by a shared belief in the glory of God and the power of prayer. I want to thank Senator Coons and Senator Lankford for the introduction and for carrying on this wonderful and uplifting bipartisan tradition."
    >>> components = parse_sentence(s)
    >>> components['speaker']
    'Donald Trump'
    >>> components['sentiment']
    'Positive'
    """
    match = sentence_regex.fullmatch(sentence)
    if match is None:
        return None

    return match.groupdict()


def parse_sentences(sentences):
    """
    >>> sentences = ["<Sean Hannity> <Positive> All right. Joining us now live on the phone is, from the White House, President Donald Trump. Mr. President, I know you've been busy. I'm sure you never predicted this as part of any presidency. Thank you for spending time with us.", "<Donald Trump> <Positive> Thank you very much, Sean.", "<Sean Hannity> <Negative> Let me stay on this issue of hydroxychloroquine. And, you know, I'm going to quote this board-certified rheumatologist for Cedars-Sinai, Dr. Wallace, who I think, obviously, he's been prescribing this now, as he pointed out, 42 years in practice. He's been the head, as one of the largest lupus practices in the U.S., currently caring for 2,000 people, most taking HCQ; 400 peer-reviewed papers; chairman of the Lupus Foundation of America, the Rheumatoid Research Foundation of America College and -- and so many other credentials.", "<Sean Hannity> <Negative> And he said, In 42 years, no patient of mine has ever been hospitalized for HCQ. And he said the risk of taking it in the doses they're talking about -- in terms of a risk, he said it is nil -- absolutely nil, and you have been getting hammered for saying: what have you got to lose? Even the AMA is saying, well, your life.", "<Sean Hannity> <Positive> I don't know. He seems pretty credible to me, sir.", "<Donald Trump> <Positive> Well, it's been taken for malaria for many years and -- very effective. It's a powerful medicine; it's a powerful drug. But it's a drug that -- for malaria, for lupus, for those two things in particular. I guess some people say arthritis, too. But it's been taken for years and people are OK with it. It seems to be, with the azithromycin -- that really seems to be the combination that's great.", "<Donald Trump> <Positive> But that could cause a little problem -- people don't know, but it might cause a problem with the heart, in which case you don't take the azithromycin. That's for infection. But the combination -- and some people add zinc. But the combination has been pretty amazing. You saw the woman state representative, a Democrat -- state representative from Michigan, Detroit.", "<Donald Trump> <Positive> And she thought she was going to die, and she saw what we were talking about and she asked her husband to get it. And she would have never known about it. And he got it and she got better. She thought she had no chance, and she got better. She -- she's been very nice about it, actually. She -- I think she may be -- might be a Democrat, but she'll vote for me, maybe.", "<Donald Trump> <Negative> But she was very nice about it. So, you know, things are happening. It's a -- it's -- I haven't seen bad. I've not seen bad. And one thing that we do see is that people are not going to die from it. So if somebody is in trouble, you take it, I think."]
    >>> sections = list(parse_sentences(sentences))
    >>> len(sections)
    4
    """
    current_speaker = None
    current_sentences = []

    for sentence in sentences:
        components = parse_sentence(sentence)
        if components is None:
            continue

        speaker = components['speaker']
        sentence = components['sentence']

        if speaker != current_speaker:
            if current_speaker is not None:
                # skip first placeholder
                yield {
                    'speaker': current_speaker,
                    'speech': ' '.join(current_sentences)
                }
            current_sentences = []
            current_speaker = speaker

        current_sentences.append(sentence)

    yield {
        'speaker': current_speaker,
        'speech': ' '.join(current_sentences)
    }


def pair_prompt_and_answer(source, answerer='Donald Trump'):
    """
    >>> sentences = ["<Sean Hannity> <Positive> All right. Joining us now live on the phone is, from the White House, President Donald Trump. Mr. President, I know you've been busy. I'm sure you never predicted this as part of any presidency. Thank you for spending time with us.", "<Donald Trump> <Positive> Thank you very much, Sean.", "<Sean Hannity> <Negative> Let me stay on this issue of hydroxychloroquine. And, you know, I'm going to quote this board-certified rheumatologist for Cedars-Sinai, Dr. Wallace, who I think, obviously, he's been prescribing this now, as he pointed out, 42 years in practice. He's been the head, as one of the largest lupus practices in the U.S., currently caring for 2,000 people, most taking HCQ; 400 peer-reviewed papers; chairman of the Lupus Foundation of America, the Rheumatoid Research Foundation of America College and -- and so many other credentials.", "<Sean Hannity> <Negative> And he said, In 42 years, no patient of mine has ever been hospitalized for HCQ. And he said the risk of taking it in the doses they're talking about -- in terms of a risk, he said it is nil -- absolutely nil, and you have been getting hammered for saying: what have you got to lose? Even the AMA is saying, well, your life.", "<Sean Hannity> <Positive> I don't know. He seems pretty credible to me, sir.", "<Donald Trump> <Positive> Well, it's been taken for malaria for many years and -- very effective. It's a powerful medicine; it's a powerful drug. But it's a drug that -- for malaria, for lupus, for those two things in particular. I guess some people say arthritis, too. But it's been taken for years and people are OK with it. It seems to be, with the azithromycin -- that really seems to be the combination that's great.", "<Donald Trump> <Positive> But that could cause a little problem -- people don't know, but it might cause a problem with the heart, in which case you don't take the azithromycin. That's for infection. But the combination -- and some people add zinc. But the combination has been pretty amazing. You saw the woman state representative, a Democrat -- state representative from Michigan, Detroit.", "<Donald Trump> <Positive> And she thought she was going to die, and she saw what we were talking about and she asked her husband to get it. And she would have never known about it. And he got it and she got better. She thought she had no chance, and she got better. She -- she's been very nice about it, actually. She -- I think she may be -- might be a Democrat, but she'll vote for me, maybe.", "<Donald Trump> <Negative> But she was very nice about it. So, you know, things are happening. It's a -- it's -- I haven't seen bad. I've not seen bad. And one thing that we do see is that people are not going to die from it. So if somebody is in trouble, you take it, I think."]
    >>> source = parse_sentences(sentences)
    >>> pairs = list(pair_prompt_and_answer(source))
    >>> len(pairs)
    2
    """
    prompt = None
    prompter = None
    for section in source:
        speaker = section['speaker']
        speech = section['speech']

        if speaker != answerer:
            # we want someone else asking answerer
            # only tend to most recent prompt
            prompt = speech
            prompter = speaker
        else:
            if prompter is None:
                # skip speech from answerer without previous prompt
                continue
            yield {
                'prompter': prompter,
                'prompt': prompt,
                'answerer': answerer,
                'answer': speech
            }
            prompt, prompter = None, None

    # ignore last unanswered prompt (if exists)


def filter_too_short_prompt(source, threshold=3):
    for pair in source:
        prompt = pair['prompt']
        if len(prompt.split()) >= threshold:
            yield pair


def filter_too_long_prompt_and_answer(source,
                                      prompt_threshold=100,
                                      answer_threshold=100):
    for pair in source:
        prompt = pair['prompt']
        answer = pair['answer']
        if len(prompt.split()) <= prompt_threshold:
            if len(answer.split()) <= answer_threshold:
                yield pair


def trim_down_prompt_and_answer(source,
                                prompt_threshold=40,
                                answer_threshold=160):
    for pair in source:
        prompt_words = pair['prompt'].split()
        if len(prompt_words) > prompt_threshold:
            pair['prompt'] = ' '.join(prompt_words[-prompt_threshold:])

        answer_words = pair['answer'].split()
        if len(answer_words) > answer_threshold:
            pair['answer'] = ' '.join(answer_words[:answer_threshold])

        yield pair


def parse_files(*filepaths, pipeline=[parse_sentences,
                                      pair_prompt_and_answer,
                                      filter_too_short_prompt,
                                      trim_down_prompt_and_answer]):
    for filepath in filepaths:
        print(f'[i] Parsing {filepath}')
        with tqdm(open(filepath)) as file_reader:

            result = file_reader
            for method in pipeline:
                result = method(result)

            yield from result


def freeze_dataset(*filepaths, target=None):
    if target is None:
        return

    dataset = list(parse_files(*filepaths))
    with open(target, 'wb') as file_writer:
        pickle.dump(dataset, file_writer)


def unfreeze_dataset(target):
    with open(target, 'rb') as file_reader:
        return pickle.load(file_reader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess dataset')
    parser.add_argument('-c', '--corpus', action='extend', nargs='+',
                        help='supply a list of corpus files')
    parser.add_argument('-s', '--save', action='store_true',
                        help='whether the generated dataset will be stored')
    parser.add_argument('target', help='filepath to dataset file')
    args = parser.parse_args()
    if args.save:
        freeze_dataset(*args.corpus, target=args.target)