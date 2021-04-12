import random

def print_true():
    print('[*] true probability')
    p_b = 6. / 10.
    p_r = 4. / 10.
    p_a_b = 3. / 4.
    p_o_b = 1. / 4.
    p_a_r = 2. / 8.
    p_o_r = 6. / 8.
    p_a = p_b * p_a_b + p_r * p_a_r
    p_o = 1. - p_a
    p_r_o = (p_o_r * p_r) / p_o
    p_b_o = 1. - p_r_o
    p_r_a = (p_a_r * p_r) / p_a
    p_b_a = 1. - p_r_a

    print('\tp(r)=', p_r)
    print('\tp(b)=', p_b)
    print('\tp(a|r)=', p_a_r)
    print('\tp(o|r)=', p_o_r)
    print('\tp(a|b)=', p_a_b)
    print('\tp(o|b)=', p_o_b)
    print('\tp(a)=', p_a)
    print('\tp(o)=', p_o)
    print('\tp(r|a)=', p_r_a)
    print('\tp(b|a)=', p_b_a)
    print('\tp(r|o)=', p_r_o)
    print('\tp(b|o)=', p_b_o)


def sample_box():
    r = random.random()
    if r < 0.4:
        return 'r'
    else:
        return 'b'

def sample_fruit_given_box(box):
    r = random.random()
    if box == 'r':
        if r < 2./8.:
            return 'a'
        else:
            return 'o'
    else:
        if r < 3./4.:
            return 'a'
        else:
            return 'o'

def main():
    itr = 10000
    count_r = 0
    count_b = 0
    count_a_r = 0
    count_a_b = 0
    count_o_r = 0
    count_o_b = 0

    for i in range(itr):
        box = sample_box()
        if box == 'r': count_r += 1
        else: count_b += 1

        fruit = sample_fruit_given_box(box)
        if box == 'r' and fruit == 'a': count_a_r += 1
        elif box == 'r' and fruit == 'o': count_o_r += 1
        elif box == 'b' and fruit == 'a': count_a_b += 1
        else: count_o_b += 1

    print('[*] estimate from sampling')
    print('\tp(r)=', float(count_r) / float(itr))
    print('\tp(b)=', float(count_b) / float(itr))
    print('\tp(a|r)=', float(count_a_r) / float(count_r))
    print('\tp(o|r)=', float(count_o_r) / float(count_r))
    print('\tp(a|b)=', float(count_a_b) / float(count_b))
    print('\tp(o|b)=', float(count_o_b) / float(count_b))
    print('\tp(a)=', float(count_a_b + count_a_r) / float(itr))
    print('\tp(o)=', float(count_o_b + count_o_r) / float(itr))
    print('\tp(r|a)=', float(count_a_r) / float(count_a_r + count_a_b))
    print('\tp(b|a)=', float(count_a_b) / float(count_a_r + count_a_b))
    print('\tp(r|o)=', float(count_o_r) / float(count_o_r + count_o_b))
    print('\tp(b|o)=', float(count_o_b) / float(count_o_r + count_o_b))

    print_true()

if __name__ == '__main__':
    main()

