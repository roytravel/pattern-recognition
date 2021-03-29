# 크리스토퍼 비숍(패턴인식과 머신러닝) 책 예제 사용


# prior
p_b = 6 / 10. #파란 상자 확률
p_r = 1 - p_b #빨간 상자 확률

# likelihood
p_a_b = 3 / 4. # 파란상자에서 사과가 나올 확률 
p_o_b = 1 - p_a_b #파란상자에서 오렌지가 나올 확률 = 1/4
p_a_r = 2 / 8. #빨간상자에서 사과가 나올 확률
p_o_r = 1. - p_a_r #빨간상자에서 오렌지가 나올 확률 = 6/8

# prior (marginal prob)
p_a = p_b * p_a_b + p_r * p_a_r
print ('p(a) =', p_a)

p_o = 1. - p_a
print ('p(o)=', p_o)

# posterior
p_r_o = (p_o_r * p_r) / p_o
print ('p(r|o)=', p_r_o)
p_b_o = 1. - p_r_o
print ('p(b|o)=', p_b_o)

p_r_a = (p_a_r * p_r) / p_a
print('p(r|a)=', p_r_a)
p_b_a = 1. - p_r_a
print ('p(b|a)', p_b_a)