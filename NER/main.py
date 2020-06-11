from data import build_corpus
from evaluate import crf_train_eval


def main():
    """训练模型，评估结果"""

    text = '''
        ####
        没有使用老师提供的数据集，O标签太多（占比92.77%），模型训练效果不好
        新数据集取自
        https://github.com/luopeixiang/named_entity_recognition
        ####'''
    print(text, '\n')
    # 读取数据
    print("读取数据...\n")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估CRF模型
    print("训练并评估CRF模型...\n")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )


if __name__ == '__main__':
    main()