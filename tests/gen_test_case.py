import random


def generate_test_case():
    """
    生成测试用例以检查Rust中的IndexMap结构与Python中的dict结构是否一致。
    注意：相关测试用例在tokenizer.rs文件中。
    """
    stats = {}
    for i in range(20):
        if random.random() < 0.25:
            value = 99
        else:
            value = random.randint(0, 20)
        stats[(i, i)] = value
        print(f"(({i},{i}), {value})")

    pair = max(stats, key=stats.get)
    print(pair)


generate_test_case()
