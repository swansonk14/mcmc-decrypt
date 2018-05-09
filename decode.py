from __future__ import division

from collections import Counter
import copy
import csv
import numpy as np


# ALPHABET


def get_alphabet(fname='alphabet.csv'):
    with open(fname, 'r') as csvfile:
        alphabet = np.array(list(csv.reader(csvfile))[0])

    return alphabet


# TEXT MODIFICATION


def char_to_index(text, alphabet=None):
    alphabet = alphabet if alphabet is not None else get_alphabet()

    char_index_mapping = {char: index for index, char in enumerate(alphabet)}
    indices = [char_index_mapping[char] for char in text]

    return indices

def index_to_char(indices, alphabet=None):
    alphabet = alphabet if alphabet is not None else get_alphabet()

    text = [alphabet[index] for index in indices]

    return text


# DATA LOADING


def get_text(fname):
    with open(fname, 'r') as f:
        text = f.read().replace('\n', ' ')

    return text

def get_letter_probabilities(fname='letter_probabilities.csv'):
    with open(fname, 'r') as csvfile:
        P = np.array([[float(p) for p in row] for row in csv.reader(csvfile)][0])

    return P

def get_letter_transition_matrix(fname='letter_transition_matrix.csv'):
    with open(fname, 'r') as csvfile:
        M = np.array([[float(p) for p in row] for row in csv.reader(csvfile)])

    return M


# DECRYPTION


def decrypt(f, y):
    x_hat = [f[y_i] for y_i in y]
    plaintext_hat = ''.join(index_to_char(x_hat))

    return plaintext_hat


# COUNTS CLASS


class Counts:
    def __init__(self, y, alphabet):
        self.y = y
        self.first = y[0]
        self.counts = np.zeros((len(alphabet), len(alphabet)))
        for i in range(1, len(y)):
            self.counts[y[i], y[i-1]] += 1

    def decrypt_counts(self, f):
        f_inverse = np.zeros(len(f), dtype=int)
        for index1, index2 in enumerate(f):
            f_inverse[index2] = index1
        decrypted_counts = self.counts[f_inverse]
        decrypted_counts = decrypted_counts[:, f_inverse]

        return decrypted_counts


# MARKOV CHAIN MONTE CARLO


def log_p_f_y_tilde(f, y_counts, log_P, log_M):
    # Note: equal to log(p_y_f)

    log_prob = log_P[f[y_counts.first]] + \
               np.sum(np.multiply(log_M,
                                  y_counts.decrypt_counts(f)))

    return log_prob

def acceptance_prob(f, f_prime, y_counts, log_P, log_M):
    # p_f_prime / p_f = exp(log(p_f_prime)) / exp(log(p_f))
    # = exp(log(p_f_prime) - log(p_f))

    log_p_f_y_tilde_f_prime = log_p_f_y_tilde(f_prime, y_counts, log_P, log_M)
    log_p_f_y_tilde_f = log_p_f_y_tilde(f, y_counts, log_P, log_M)
    ratio = np.exp(log_p_f_y_tilde_f_prime - log_p_f_y_tilde_f)

    a = min(1, ratio)

    return a

def most_common(lst):
    counter = Counter(lst)
    most_common_element = counter.most_common()[0][0]

    return most_common_element

def mcmc(alphabet, y_counts, log_P, log_M, num_iters):
    fs = []
    f = np.random.permutation(len(alphabet))  # random initialization

    for _ in range(num_iters):
        index1, index2 = np.random.choice(len(alphabet), size=2, replace=False)
        f_prime = copy.deepcopy(f)
        f_prime[index1], f_prime[index2] = f_prime[index2], f_prime[index1]

        a = acceptance_prob(f, f_prime, y_counts, log_P, log_M)

        if np.random.rand() <= a:
            f = f_prime

        fs.append(str(f.tolist()))

    f_star = np.array(eval(most_common(fs)))
    log_likelihood = log_p_f_y_tilde(f_star, y_counts, log_P, log_M)

    return f_star, log_likelihood

def multi_mcmc(y, training_size, log_P, log_M, num_iters, num_mcmcs):
    alphabet = get_alphabet()

    start = np.random.randint(max(0, len(y)-training_size) + 1)
    training_y = y[start:start+training_size]
    y_counts = Counts(training_y, alphabet)

    best_f = None
    best_log_likelihood = float('-inf')

    for _ in range(num_mcmcs):
        f, log_likelihood = mcmc(alphabet, y_counts, log_P, log_M, num_iters)

        if log_likelihood >= best_log_likelihood:
            best_f = f
            best_log_likelihood = log_likelihood

    return best_f


# DECODE


def decode(ciphertext, output_file_name,
           training_size=1000, num_iters=10000, num_mcmc=10):
    # Convert text to indices
    y = char_to_index(ciphertext)

    # Load probabilities and convert to log domain
    log_P = np.log(get_letter_probabilities())
    log_M = np.nan_to_num(np.log(get_letter_transition_matrix()))

    # Run MCMC
    f_star = multi_mcmc(y, training_size, log_P, log_M, num_iters, num_mcmc)

    # Decrypt full ciphertext
    plaintext_hat = decrypt(f_star, y)

    # Write decryption
    with open(output_file_name, 'w') as f:
        f.write(plaintext_hat)

decode('oryabdgapyx.ybtbalye.azy.eyihzgrygcxotoxlyoxyo yrbcb  galyxigxyxibymhb xo.ry i.hnpysbyg kbpyea.zyxozbyx.yxozbyfigxyo yox yuhau. bygrpyopbgnyoryfigxyfglyp.b yoxyc.rxaoshxbyx.yxibysbghxly.eyihzgrybvo xbrcbyg yab ubcx yxi. byuha hox yfiociyc.rxaoshxby.rnlyabz.xbnlyslyua.topordyxibyzbcigro zy.eynoebyoxyo yfbnnyx.ysbyabzorpbpyxigxyr.xyxibyzbabyegcxy.eynotordyo yx.ysbypb oabpyshxyxibygaxy.eynotordyoryxibyc.rxbzungxo.ry.eydabgxyxiord jy xonnyz.abyoryabdgapyx.yxi. bygt.cgxo.r yfiociyigtbyr.ybrpy.hx opbyxibz bntb yfiociygabyx.ysbywh xoeobpyoeygxygnnyg ygcxhgnnlygppordyx.yxiby hzy.eyxibyf.anp yubazgrbrxyu.  b  o.r yoxyo yrbcb  galyx.ykbbuygnotbygykr.fnbpdby.eyxiboaygoz ygycnbgayuabeodhaordyto o.ry.eyxibyxbzunbyoryfiociycabgxotbyozgdorgxo.ryo yx.ysbybzs.pobpjyxibyehneonzbrxy.eyxio yrbbpyoryfigxyc.rcbar yxiby xhpob ye.azordyxibyzgxbaognyhu.ryfiociych x.zyig ypbcopbpyx.yxagoryxibyl.hxiehnyzorpyo yorpbbpy gpnlyabz.xb .yabz.xbyg yx.yzgkbyxibyzbaby xgxbzbrxy.ey hciygycngozyguubgayuabu. xba.h jydabgxyzbryehnnlygnotbyx.yxibysbghxly.eyxibyc.rxbzungxo.r yx.yfi. by batocbyxiboaynotb ygabypbt.xbpypb oaordyxigxy.xiba yzgly igabyoryxiboayw.l yuba hgpbyzgrkorpyx.yozugaxyx.yxiby hccb  otbydbrbagxo.r yxibyzbcigrocgnykr.fnbpdbyfoxi.hxyfiociyoxyo yozu.  osnbyx.yca.  yxibyxiab i.npjypalyubpgrx yu.  b  yxibz bntb y.eyxibyuaotonbdby.eyor xonnordyxio ykr.fnbpdbyxiblye.adbxyxigxyoxyo yx.y batbyshxyg ygykblyx.y.ubryxibyp..a y.eyxibyxbzunbyxi.hdiyxibly ubrpyxiboaynotb y.ryxiby xbu ynbgpordyhuyx.yxi. by gcabpyp..a yxiblyxharyxiboaysgck yhu.ryxibyxbzunby .yab .nhxbnlyxigxyox ytbalybvo xbrcbyo ye.ad.xxbrygrpyxibybgdbayl.hxiyfi.yf.hnpyuab  ye.afgapyx.ysbyoroxogxbpyx.yox yp.zb ygrpygacib yo ysoppbryx.yxharysgckygrpyc.hrxyxiby xbu jyzgxibzgxoc yubaigu yz.abybtbryxigryxiby xhply.eydabbcbygrpya.zbyig y heebabpyea.zyxio y.snoto.ry.eyox yphbyungcbyorycotono gxo.rjygnxi.hdiyxagpoxo.ryig ypbcabbpyxigxyxibydabgxyshnky.eybphcgxbpyzbry ignnykr.fygxynbg xyxibybnbzbrx y.eyxiby hswbcxyxibyabg .r ye.ayfiociyxibyxagpoxo.ryga. bygabye.ad.xxbryshaobpysbrbgxiygydabgxyahsso iibguy.eyubpgrxaob ygrpyxaotognoxob jyx.yxi. byfi.yormhoabyg yx.yxibyuhau. by.eyzgxibzgxoc yxibyh hgnygr fbayfonnysbyxigxyoxyegconoxgxb yxibyzgkordy.eyzgciorb yxibyxagtbnnordyea.zyungcbyx.yungcbygrpyxibytocx.aly.tbaye.abodryrgxo.r yfibxibayoryfgay.ayc.zzbacbjyoeyoxysby.swbcxbpyxigxyxib bybrp gnny.eyfiociygaby.eyp.hsxehnytgnhbgabyr.xyehaxibabpyslyxibyzbabnlybnbzbrxgaly xhplyozu. bpyhu.ryxi. byfi.yp.yr.xysbc.zbybvubaxyzgxibzgxocogr yxibyabunlyoxyo yxahbyfonnyua.sgsnlysbyxigxyzgxibzgxoc yxagor yxibyabg .rordyegchnxob jylbxyxibytbalyzbryfi.yzgkbyxio yabunlygabye.ayxibyz. xyugaxyhrfonnordyx.ygsgrp.ryxibyxbgciordy.eypbeoroxbyegnngcob ykr.fryx.ysby hciygrpyor xorcxotbnlyabwbcxbpyslyxibyhr .uio xocgxbpyzorpy.eybtbalyorxbnnodbrxynbgarbajygrpyxibyabg .rordyegchnxlyox bneyo ydbrbagnnlyc.rcbotbpyslyxi. byfi.yhadbyox ychnxotgxo.ryg yzbabnlygyzbgr ye.ayxibygt.opgrcby.eyuoxegnn ygrpygyibnuyoryxibypo c.tbaly.eyahnb ye.ayxibydhopgrcby.eyuagcxocgnynoebjygnnyxib bygabyhrpbrogsnlyozu.axgrxygciobtbzbrx yx.yxibycabpoxy.eyzgxibzgxoc ylbxyoxyo yr.rby.eyxib byxigxybrxoxnb yzgxibzgxoc yx.ygyungcbyorybtbalynosbagnybphcgxo.rjyungx.yfbykr.fyabdgapbpyxibyc.rxbzungxo.ry.eyzgxibzgxocgnyxahxi yg yf.axily.eyxibypboxlygrpyungx.yabgno bpyz.abyubaigu yxigrygrly.xibay ordnbyzgryfigxyxi. bybnbzbrx ygabyoryihzgrynoebyfiociyzbaoxygyungcbyoryibgtbrjyxibabyo yoryzgxibzgxoc yiby gl y .zbxiordyfiociyo yrbcb  galygrpycgrr.xysby bxyg opbygrpyoeyoyzo xgkbyr.xy.eypotorbyrbcb  oxlye.ayg yx.yxibyihzgryrbcb  oxob y.eyfiociyxibyzgrlyxgnkyoryxio yc.rrbcxo.ryr.xiordycgrysbyz.abyaopochn.h yxigry hciygryguunocgxo.ry.eyxibyf.ap jycnborog jygrpyfigxygabyxib byrbcb  oxob y.eykr.fnbpdby xagrdbayfiociygabypotorbygrpyr.xyihzgrygxibrogrjyxi. byxiord yfoxi.hxy .zbyh by.aykr.fnbpdby.eyfiociygyzgrycgrr.xysbc.zbygyd.pyx.yxibyf.anpyr.aygy uoaoxyr.aylbxygyiba.yr.aygsnbybgarb xnlyx.yxiorkygrpycgabye.ayzgryngf yujy hciyfg yungx. ywhpdzbrxy.eyzgxibzgxoc yshxyxibyzgxibzgxocogr yp.yr.xyabgpyungx.yfionbyxi. byfi.yabgpyiozykr.fyr.yzgxibzgxoc ygrpyabdgapyio y.uoro.ryhu.ryxio ymhb xo.ryg yzbabnlygychao.h ygsbaagxo.rjyzgxibzgxoc yaodixnlytobfbpyu.  b  b yr.xy.rnlyxahxiyshxy huabzbysbghxlgysbghxlyc.npygrpygh xbabynokbyxigxy.ey chnuxhabyfoxi.hxyguubgnyx.ygrlyugaxy.ey.hayfbgkbayrgxhabyfoxi.hxyxibyd.adb.h yxaguuord y.eyugorxordy.ayzh ocylbxy hsnozbnlyuhabygrpycgugsnby.eygy xbaryubaebcxo.ry hciyg y.rnlyxibydabgxb xygaxycgry i.fjyxibyxahby uoaoxy.eypbnodixyxibybvgnxgxo.ryxiby br by.eysbordyz.abyxigryzgryfiociyo yxibyx.hci x.rby.eyxibyiodib xybvcbnnbrcbyo yx.ysbye.hrpyoryzgxibzgxoc yg y habnlyg yoryu.bxaljyfigxyo ysb xyoryzgxibzgxoc ypb batb yr.xyzbabnlyx.ysbynbgarxyg ygyxg kyshxyx.ysbyg  ozongxbpyg ygyugaxy.eypgonlyxi.hdixygrpysa.hdixygdgorygrpygdgorysbe.abyxibyzorpyfoxiybtbaabrbfbpybrc.hagdbzbrxjyabgnynoebyo yx.yz. xyzbrygyn.rdy bc.rpsb xygyubaubxhgnyc.zua.zo bysbxfbbryxibyopbgnygrpyxibyu.  osnbyshxyxibyf.anpy.eyuhabyabg .rykr.f yr.yc.zua.zo byr.yuagcxocgnynozoxgxo.r yr.ysgaaobayx.yxibycabgxotbygcxotoxlybzs.plordyory unbrpopybpoeocb yxibyug  o.rgxbyg uoagxo.rygexbayxibyubaebcxyea.zyfiociygnnydabgxyf.aky uaord jyabz.xbyea.zyihzgryug  o.r yabz.xbybtbryea.zyxibyuoxoehnyegcx y.eyrgxhabyxibydbrbagxo.r yigtbydagphgnnlycabgxbpygry.apbabpyc. z. yfibabyuhabyxi.hdixycgrypfbnnyg yoryox yrgxhagnyi.zbygrpyfibaby.rbygxynbg xy.ey.hayr.snbayozuhn b ycgryb cgubyea.zyxibypabgalybvonby.eyxibygcxhgnyf.anpjy .ynoxxnbyi.fbtbayigtbyzgxibzgxocogr ygozbpygxysbghxlyxigxyigapnlygrlxiordyoryxiboayf.akyig yigpyxio yc.r co.h yuhau. bjyzhciy.fordyx.yoaabuab  osnbyor xorcx yfiociyfbabysbxxbayxigrygt.fbpysbnobe yig ysbbryz.hnpbpyslygryhrc.r co.h yxg xbyshxyzhciygn .yig ysbbry u.onxyslyegn byr.xo.r y.eyfigxyfg yeoxxordjyxibycigagcxbao xocybvcbnnbrcby.eyzgxibzgxoc yo y.rnlyx.ysbye.hrpyfibabyxibyabg .rordyo yaodopnlyn.docgnyxibyahnb y.eyn.docygabyx.yzgxibzgxoc yfigxyxi. by.ey xahcxhabygabyx.ygacioxbcxhabjyoryxibyz. xysbghxoehnyf.akygycigory.eygadhzbrxyo yuab brxbpyoryfiociybtbalynorkyo yozu.axgrxy.ryox y.frygcc.hrxyoryfiociyxibabyo ygrygoay.eybg bygrpynhcopoxlyxia.hdi.hxygrpyxibyuabzo b ygciobtbyz.abyxigryf.hnpyigtbysbbryxi.hdixyu.  osnbyslyzbgr yfiociyguubgayrgxhagnygrpyorbtoxgsnbjynoxbagxhabybzs.pob yfigxyo ydbrbagnyoryugaxochngaycoachz xgrcb yfi. byhrotba gny odroeocgrcby iorb yxia.hdiyxiboayorpotophgnypab  yshxyzgxibzgxoc ybrpbgt.ha yx.yuab brxyfigxbtbayo yz. xydbrbagnyoryox yuhaoxlyfoxi.hxygrlyoaabnbtgrxyxaguuord jyi.fy i.hnpyxibyxbgciordy.eyzgxibzgxoc ysbyc.rphcxbpy .yg yx.yc.zzhrocgxbyx.yxibynbgarbayg yzhciyg yu.  osnby.eyxio yiodiyopbgnyibabybvubaobrcbyzh xyorygydabgxyzbg habysby.haydhopbyshxy .zbyzgvoz yzglyab hnxyea.zy.hayc.r opbagxo.ry.eyxibyhnxozgxbyuhau. byx.ysbygciobtbpjy.rby.eyxibyciobeybrp y batbpyslyzgxibzgxoc yfibryaodixnlyxghdixyo yx.ygfgkbryxibynbgarba ysbnobeyoryabg .ryio yc.reopbrcbyoryxibyxahxiy.eyfigxyig ysbbrypbz.r xagxbpygrpyoryxibytgnhby.eypbz.r xagxo.rjyxio yuhau. byo yr.xy batbpyslybvo xordyor xahcxo.ryshxyoxyo ybg lyx.y bbyfgl yoryfiociyoxyzodixysby batbpjygxyuab brxyoryfigxyc.rcbar ygaoxizbxocyxibys.ly.aydoanyo ydotbrygy bxy.eyahnb yfiociyuab brxyxibz bntb yg yrboxibayxahbyr.ayegn byshxyg yzbabnlyxibyfonny.eyxibyxbgcibayxibyfglyoryfiociye.ay .zbyhregxi.zgsnbyabg .ryxibyxbgcibayuabeba yx.yigtbyxibydgzbyunglbpjyx.y .zbypbdabbyorygy xhply.ey hciypbeoroxbyuagcxocgnyhxonoxlyxio yo yr.yp.hsxyhrgt.opgsnbyshxyg y ..ryg yu.  osnbyxibyabg .r y.eyahnb y i.hnpysby bxye.axiyslyfigxbtbayzbgr yz. xyabgponlyguubgnyx.yxibycionpo iyzorpjyorydb.zbxalyor xbgpy.eyxibyxbpo.h yguugagxh y.eyegnngco.h yua..e ye.ay.sto.h yxaho z yfiociyc.r xoxhxb yxibysbdorrordy.eybhcnopyxibynbgarbay i.hnpysbygnn.fbpygxyeoa xyx.yg  hzbyxibyxahxiy.eybtbalxiordy.sto.h ygrpy i.hnpysbyor xahcxbpyoryxibypbz.r xagxo.r y.eyxib.abz yfiociygabygxy.rcby xgaxnordygrpybg onlytbaoeogsnbyslygcxhgnypagfordy hciyg yxi. byoryfiociyoxyo y i.fryxigxyxiabby.ayz.abynorb yzbbxyorygyu.orxjyoryxio yfglysbnobeyo ydbrbagxbpyoxyo y bbryxigxyabg .rordyzglynbgpyx.y xgaxnordyc.rcnh o.r yfiociyrbtbaxibnb  yxibyegcx yfonnytbaoelygrpyxih yxibyor xorcxotbypo xah xy.eyfigxbtbayo ygs xagcxy.ayagxo.rgnyo ydagphgnnly.tbac.zbjyfibabyxib.abz ygabypoeeochnxyxibly i.hnpysbyeoa xyxghdixyg ybvbaco b yorydb.zbxaocgnypagfordyhrxonyxibyeodhabyig ysbc.zbyxi.a.hdinlyegzonogayoxyfonnyxibrysbygrygdabbgsnbygptgrcbyx.ysbyxghdixyxibyn.docgnyc.rrbcxo.r y.eyxibytgao.h ynorb y.aycoacnb yxigxy.cchajyoxyo ypb oagsnbygn .yxigxyxibyeodhabyonnh xagxordygyxib.abzy i.hnpysbypagfryorygnnyu.  osnbycg b ygrpy igub yxigxy .yxibygs xagcxyabngxo.r yfoxiyfiociydb.zbxalyo yc.rcbarbpyzgly.eyxibz bntb ybzbadbyg yxibyab ophby.ey ozongaoxlygzopy hciydabgxyguugabrxypotba oxljyoryxio yfglyxibygs xagcxypbz.r xagxo.r y i.hnpye.azyshxygy zgnnyugaxy.eyxibyor xahcxo.rygrpy i.hnpysbydotbryfibryslyegzonogaoxlyfoxiyc.rcabxbyonnh xagxo.r yxiblyigtbyc.zbyx.ysbyebnxyg yxibyrgxhagnybzs.pozbrxy.eyto osnbyegcxjyoryxio ybganly xgdbyua..e y i.hnpyr.xysbydotbryfoxiyubpgrxocyehnnrb  ypbeoroxbnlyegnngco.h yzbxi.p y hciyg yxigxy.ey hubau. oxo.ry i.hnpysbyaodopnlybvcnhpbpyea.zyxibyeoa xyshxyfibabyfoxi.hxy hciyzbxi.p yxibyua..eyf.hnpysbytbalypoeeochnxyxibyab hnxy i.hnpysbyabrpbabpygccbuxgsnbyslygadhzbrx ygrpyonnh xagxo.r yfiociygabybvunocoxnlyc.rxag xbpyfoxiypbz.r xagxo.r jyoryxibysbdorrordy.eygndbsagybtbryxibyz. xyorxbnnodbrxycionpyeorp yg ygyahnbytbalydabgxypoeeochnxljyxibyh by.eynbxxba yo ygyzl xbalyfiociy bbz yx.yigtbyr.yuhau. bybvcbuxyzl xoeocgxo.rjyoxyo ygnz. xyozu.  osnbygxyeoa xyr.xyx.yxiorkyxigxybtbalynbxxbay xgrp ye.ay .zbyugaxochngayrhzsbayoey.rnlyxibyxbgcibayf.hnpyabtbgnyfigxyrhzsbayoxy xgrp ye.ajyxibyegcxyo yxigxyorygndbsagyxibyzorpyo yeoa xyxghdixyx.yc.r opbaydbrbagnyxahxi yxahxi yfiociygabyr.xyg  baxbpyx.yi.npy.rnly.eyxio y.ayxigxyugaxochngayxiordyshxy.eygrly.rby.eygyfi.nbyda.huy.eyxiord jyoxyo yoryxibyu.fbay.eyhrpba xgrpordygrpypo c.tbaordy hciyxahxi yxigxyxibyzg xbaly.eyxibyorxbnnbcxy.tbayxibyfi.nbyf.anpy.eyxiord ygcxhgnygrpyu.  osnbyab opb ygrpygsonoxlyx.ypbgnyfoxiyxibydbrbagnyg y hciyo y.rby.eyxibydoex yxigxygyzgxibzgxocgnybphcgxo.ry i.hnpysb x.fjyshxyi.fynoxxnbyg ygyahnbyo yxibyxbgcibay.eygndbsagygsnbyx.ybvungoryxibycig zyfiociypotopb yoxyea.zygaoxizbxocygrpyi.fynoxxnbyo yxibynbgarbayg  o xbpyoryio yda.uordybee.ax ygxyc.zuabibr o.ryh hgnnlyxibyzbxi.pyxigxyig ysbbrygp.uxbpyorygaoxizbxocyo yc.rxorhbpyahnb ygaby bxye.axiyfoxiyr.ygpbmhgxbybvungrgxo.ry.eyxiboayda.hrp yxibyuhuonynbgar yx.yh byxibyahnb ysnorpnlygrpyuab brxnlyfibryibyo ygsnbyx.y.sxgoryxibygr fbayxigxyxibyxbgcibaypb oab yibyebbn yxigxyibyig yzg xbabpyxibypoeeochnxob y.eyxiby hswbcxjyshxy.eyorrbayc.zuabibr o.ry.eyxibyua.cb  b ybzun.lbpyibyig yua.sgsnlygcmhoabpygnz. xyr.xiordjyfibrygndbsagyig ysbbrynbgarxygnnyd.b y z..xinlyhrxonyfbyabgciyxi. by xhpob yoryfiociyxibyr.xo.ry.eyoreoroxlyo ybzun.lbpxibyoreoroxb ozgnycgnchnh ygrpyxibyfi.nby.eyiodibayzgxibzgxoc jyxiby .nhxo.ry.eyxibypoeeochnxob yfiociye.azbanly haa.hrpbpyxibyzgxibzgxocgnyoreoroxbyo yua.sgsnlyxibydabgxb xygciobtbzbrxy.eyfiociy.hay.frygdbyig yx.ys.g xjy orcbyxibysbdorrord y.eydabbkyxi.hdixyxib bypoeeochnxob yigtbysbbrykr.fryorybtbalygdbyxibyeorb xyorxbnnbcx yigtbytgornlybrpbgt.habpyx.ygr fbayxibyguugabrxnlyhrgr fbagsnbymhb xo.r yxigxyigpysbbryg kbpyslyqbr.yxibybnbgxocjygxyng xydb.adycgrx.ayig ye.hrpyxibygr fbaygrpyig yc.rmhbabpye.ayxibyorxbnnbcxygyrbfygrpytg xyua.torcbyfiociyigpysbbrydotbry.tbayx.ycig. ygrpy.npyrodixjyoxyfg yg  hzbpyg y bnebtopbrxyhrxonycgrx.aygrpypbpbkorpyb xgsno ibpyxiby.uu. oxbyxigxyoeyea.zygrlyc.nnbcxo.ry.eyxiord y .zbyfbabyxgkbrygfglyxibyrhzsbay.eyxiord ynbexyzh xygnfgl ysbynb  yxigryxiby.aodorgnyrhzsbay.eyxiord jyxio yg  hzuxo.ryg ygyzgxxbay.eyegcxyi.np y.rnly.eyeoroxbyc.nnbcxo.r ygrpyxibyabwbcxo.ry.eyoxyfibabyxibyoreoroxbyo yc.rcbarbpyig ysbbry i.fryx.yabz.tbygnnyxibypoeeochnxob yxigxyigpyioxibax.ysgeenbpyihzgryabg .ryoryxio yzgxxbaygrpyx.yabrpbayu.  osnbyxibycabgxo.ry.eygrybvgcxy cobrcby.eyxibyoreoroxbjyxio y xhubrp.h yegcxy.hdixyx.yua.phcbygyabt.nhxo.ryoryxibyiodibayxbgciordy.eyzgxibzgxoc yoxyig yox bneygppbpyozzbg hagsnlyx.yxibybphcgxo.rgnytgnhby.eyxiby hswbcxygrpyoxyig ygxyng xydotbryxibyzbgr y.eyxabgxordyfoxiyn.docgnyuabco o.ryzgrly xhpob yfiociyhrxonyngxbnlyfbabyfaguubpyoryegnngclygrpy.s chaoxljyslyxi. byfi.yfbabybphcgxbpy.ryxiby.npynorb yxibyrbfyf.akyo yc.r opbabpyx.ysbyguugnnordnlypoeeochnxygs xah bygrpy.s chabygrpyoxyzh xysbyc.reb  bpyxigxyxibypo c.tbabayg yo y .y.exbryxibycg byig yigapnlyioz bneybzbadbpyea.zyxibyzo x yfiociyxibynodixy.eyio yorxbnnbcxyo ypo ubnnordjyshxyoribabrxnlyxibyrbfyp.cxaorby.eyxibyoreoroxbyx.ygnnycgrpopygrpyormhoaordyzorp yig yegconoxgxbpyxibyzg xbaly.eyiodibayzgxibzgxoc ye.ayioxibax.yoxyig ysbbryrbcb  galyx.ynbgaryslygyn.rdyua.cb  y.ey .uio xocgxo.ryx.ydotbyg  brxyx.ygadhzbrx yfiociy.ryeoa xygcmhgorxgrcbyfbabyaodixnlywhpdbpyx.ysbyc.reh bpygrpybaa.rb.h jy .yegayea.zyua.phcordygyebganb  ysbnobeyoryabg .rygys.npyabwbcxo.ry.eyfigxbtbayegonbpyx.yehneonyxiby xaocxb xyabmhoabzbrx y.eyn.docygyzgxibzgxocgnyxagorordyphaordyxibyug xyxf.ycbrxhaob ybrc.hagdbpyxibysbnobeyxigxyzgrlyxiord yfiociygyaodopyormhoalyf.hnpyabwbcxyg yegnngco.h yzh xylbxysbygccbuxbpysbcgh byxiblyf.akyoryfigxyxibyzgxibzgxocogrycgnn yuagcxocbjyslyxio yzbgr ygyxozopyc.zua.zo ordy uoaoxy.aybn bygy gcbap.xgnysbnobeyoryzl xbaob yr.xyorxbnnodosnbyx.yxibyua.egrbyig ysbbrysabpyfibabyabg .rygn.rby i.hnpyigtbyahnbpjygnnyxio yoxyo yr.fyxozbyx.y fbbuygfglynbxyxi. byfi.yfo iyx.yubrbxagxbyorx.yxibygacgrgy.eyzgxibzgxoc ysbyxghdixygxy.rcbyxibyxahbyxib.alyorygnnyox yn.docgnyuhaoxlygrpyoryxibyc.rcgxbrgxo.ryb xgsno ibpyslyxibytbalyb  brcby.eyxibybrxoxob yc.rcbarbpjyoeyfbygabyc.r opbaordyzgxibzgxoc yg ygrybrpyoryox bneygrpyr.xyg ygyxbcirocgnyxagorordye.aybrdorbba yoxyo ytbalypb oagsnbyx.yuab batbyxibyuhaoxlygrpy xaocxrb  y.eyox yabg .rordjygcc.apordnlyxi. byfi.yigtbygxxgorbpygy heeocobrxyegzonogaoxlyfoxiyox ybg obayu.axo.r y i.hnpysbynbpysgckfgapyea.zyua.u. oxo.r yx.yfiociyxiblyigtbyg  brxbpyg y bnebtopbrxyx.yz.abygrpyz.abyehrpgzbrxgnyuaorcounb yea.zyfiociyfigxyigpyuabto.h nlyguubgabpyg yuabzo b ycgrysbypbphcbpjyxibly i.hnpysbyxghdixfigxyxibyxib.aly.eyoreoroxlytbalyguxnlyonnh xagxb xigxyzgrlyua.u. oxo.r y bbzy bnebtopbrxyx.yxibyhrxagorbpyzorpyfiociyrbtbaxibnb  ygyrbgabay cahxorly i.f yx.ysbyegn bjyslyxio yzbgr yxiblyfonnysbynbpyx.ygy cbuxocgnyormhoalyorx.yeoa xyuaorcounb ygrybvgzorgxo.ry.eyxibye.hrpgxo.r yhu.ryfiociyxibyfi.nbybpoeocby.eyabg .rordyo yshonxy.ayx.yxgkbyubaigu ygyz.abyeoxxordyzbxgui.ayxibydabgxyxahrkyea.zyfiociyxiby uabgpordysagrcib y uaordjygxyxio y xgdbyoxyo yfbnnyx.y xhplygeab iyxibybnbzbrxgalyu.axo.r y.eyzgxibzgxoc yg kordyr.yn.rdbayzbabnlyfibxibaygydotbryua.u. oxo.ryo yxahbyshxygn .yi.fyoxyda.f y.hxy.eyxibycbrxagnyuaorcounb y.eyn.docjymhb xo.r y.eyxio yrgxhabycgryr.fysbygr fbabpyfoxiygyuabco o.rygrpycbaxgorxlyfiociyfbabye.azbanlymhoxbyozu.  osnbygrpyoryxibycigor y.eyabg .rordyxigxyxibygr fbayabmhoab yxibyhroxly.eygnnyzgxibzgxocgny xhpob ygxyng xyhre.np yox bnejyoryxibydabgxyzgw.aoxly.eyzgxibzgxocgnyxbvxs..k yxibabyo ygyx.xgnyngcky.eyhroxlyoryzbxi.pygrpy.ey l xbzgxocypbtbn.uzbrxy.eygycbrxagnyxibzbjyua.u. oxo.r y.eytbalypotba bykorp ygabyua.tbpyslyfigxbtbayzbgr ygabyxi.hdixyz. xybg onlyorxbnnodosnbygrpyzhciy ugcbyo ypbt.xbpyx.yzbabychao. oxob yfiociyoryr.yfglyc.rxaoshxbyx.yxibyzgorygadhzbrxjyshxyoryxibydabgxb xyf.ak yhroxlygrpyorbtoxgsonoxlygabyebnxyg yoryxibyhre.npordy.eygypagzgyoryxibyuabzo  b ygy hswbcxyo yua.u. bpye.ayc.r opbagxo.rygrpyorybtbaly hs bmhbrxy xbuy .zbypbeoroxbygptgrcbyo yzgpbyx.fgap yzg xbaly.eyox yrgxhabjyxibyn.tby.ey l xbzy.eyorxbac.rrbcxo.ryfiociyo yubaigu yxibyorz. xyb  brcby.eyxibyorxbnnbcxhgnyozuhn bycgryeorpyeabbyunglyoryzgxibzgxoc yg yr.fibabybn bjyxibynbgarbayfi.yebbn yxio yozuhn byzh xyr.xysbyabubnnbpyslygrygaagly.eyzbgrordnb  ybvgzunb y.aypo xagcxbpyslygzh ordy.ppoxob yshxyzh xysbybrc.hagdbpyx.ypfbnnyhu.rycbrxagnyuaorcounb yx.ysbc.zbyegzonogayfoxiyxiby xahcxhaby.eyxibytgao.h y hswbcx yfiociygabyuhxysbe.abyiozyx.yxagtbnybg only.tbayxiby xbu y.eyxibyz.abyozu.axgrxypbphcxo.r jyoryxio yfglygyd..pyx.rby.eyzorpyo ychnxotgxbpygrpy bnbcxotbygxxbrxo.ryo yxghdixyx.ypfbnnyslyuabebabrcbyhu.ryfigxyo yfbodixlygrpyb  brxognjyfibryxiby bugagxby xhpob yorx.yfiociyzgxibzgxoc yo ypotopbpyigtbybgciysbbrytobfbpyg ygyn.docgnyfi.nbyg ygyrgxhagnyda.fxiyea.zyxibyua.u. oxo.r yfiociyc.r xoxhxbyxiboayuaorcounb yxibynbgarbayfonnysbygsnbyx.yhrpba xgrpyxibyehrpgzbrxgny cobrcbyfiociyhroeob ygrpy l xbzgxo b yxibyfi.nby.eypbphcxotbyabg .rordjyxio yo y lzs.nocyn.docgy xhplyfiociyxi.hdiyoxy.fb yox yorcbuxo.ryx.ygao x.xnbyo ylbxyoryox yfopbaypbtbn.uzbrx ygyua.phcxygnz. xyfi.nnly.eyxibyrorbxbbrxiycbrxhalygrpyo yorpbbpyoryxibyuab brxypgly xonnyda.fordyfoxiydabgxyaguopoxljyxibyxahbyzbxi.py.eypo c.tbalyory lzs.nocyn.docygrpyua.sgsnlygn .yxibysb xyzbxi.pye.ayorxa.phcordyxiby xhplyx.ygynbgarbaygcmhgorxbpyfoxiy.xibayugax y.eyzgxibzgxoc yo yxibygrgnl o y.eygcxhgnybvgzunb y.eypbphcxotbyabg .rordyfoxiygytobfyx.yxibypo c.tbaly.eyxibyuaorcounb ybzun.lbpjyxib byuaorcounb ye.ayxibyz. xyugaxygaby .ybzsbppbpyory.hayagxo.corgxotbyor xorcx yxigxyxiblygabybzun.lbpymhoxbyhrc.r co.h nlygrpycgrysbypagddbpyx.ynodixy.rnlyslyzhciyugxobrxybee.axjyshxyfibrygxyng xyxiblyigtbysbbrye.hrpyxiblygaby bbryx.ysbyebfyoryrhzsbaygrpyx.ysbyxiby .nby .hacby.eybtbalxiordyoryuhabyzgxibzgxoc jyxibypo c.tbalyxigxygnnyzgxibzgxoc ye.nn.f yorbtoxgsnlyea.zygy zgnnyc.nnbcxo.ry.eyehrpgzbrxgnyngf yo y.rbyfiociyozzbg hagsnlybrigrcb yxibyorxbnnbcxhgnysbghxly.eyxibyfi.nbyx.yxi. byfi.yigtbysbbry.uuab  bpyslyxibyeagdzbrxgalygrpyorc.zunbxbyrgxhaby.eyz. xybvo xordycigor y.eypbphcxo.ryxio ypo c.tbalyc.zb yfoxiygnnyxiby.tbafibnzordye.acby.eygyabtbngxo.rynokbygyugngcbybzbadordyea.zyxibyghxhzryzo xyg yxibyxagtbnnbayg cbrp ygryoxgnogryionn opbyxiby xgxbnly x.abl y.eyxibyzgxibzgxocgnybpoeocbyguubgayoryxiboayphby.apbaygrpyua.u.axo.ryfoxiygyrbfyubaebcxo.ryorybtbalyugaxjyhrxony lzs.nocyn.docyigpygcmhoabpyox yuab brxypbtbn.uzbrxyxibyuaorcounb yhu.ryfiociyzgxibzgxoc ypbubrp yfbabygnfgl y huu. bpyx.ysbyuion. .uiocgnygrpypo c.tbagsnby.rnlyslyxibyhrcbaxgoryhrua.dab  otbyzbxi.p yioxibax.ybzun.lbpyslyuion. .uiba jy .yn.rdyg yxio yfg yxi.hdixyzgxibzgxoc y bbzbpyx.ysbyr.xyghx.r.z.h yshxypbubrpbrxyhu.rygy xhplyfiociyigpymhoxby.xibayzbxi.p yxigryox y.frjyz.ab.tbay orcbyxibyrgxhaby.eyxibyu. xhngxb yea.zyfiociygaoxizbxocygrgnl o ygrpydb.zbxalygabyx.ysbypbphcbpyfg yfaguubpyorygnnyxibyxagpoxo.rgny.s chaoxob y.eyzbxguil ocgnypo ch  o.ryxibybpoeocbyshonxyhu.ry hciyphso.h ye.hrpgxo.r ysbdgryx.ysbytobfbpyg yr.ysbxxbayxigrygycg xnbyoryxibygoajyoryxio yab ubcxyxibypo c.tbalyxigxyxibyxahbyuaorcounb ygabyg yzhciygyugaxy.eyzgxibzgxoc yg ygrly.eyxiboayc.r bmhbrcb yig ytbalydabgxnlyorcabg bpyxibyorxbnnbcxhgny gxo egcxo.ryx.ysby.sxgorbpjyxio y gxo egcxo.ry.hdixyr.xyx.ysbyabeh bpyx.ynbgarba ycgugsnby.eybrw.lordyoxye.ayoxyo y.eygykorpyx.yorcabg by.hayab ubcxye.ayihzgryu.fba ygrpy.haykr.fnbpdby.eyxibysbghxob ysbn.rdordyx.yxibygs xagcxyf.anpjyuion. .uiba yigtbyc.zz.rnlyibnpyxigxyxibyngf y.eyn.docyfiociyhrpbanobyzgxibzgxoc ygabyngf y.eyxi.hdixyngf yabdhngxordyxiby.ubagxo.r y.ey.hayzorp jyslyxio y.uoro.ryxibyxahbypodroxly.eyabg .ryo ytbalydabgxnlyn.fbabpyoxycbg b yx.ysbygryortb xodgxo.ryorx.yxibytbalyibgaxygrpyozzhxgsnbyb  brcby.eygnnyxiord ygcxhgnygrpyu.  osnbysbc.zordyor xbgpygryormhoalyorx.y .zbxiordyz.aby.aynb  yihzgrygrpy hswbcxyx.y.haynozoxgxo.r jyxibyc.rxbzungxo.ry.eyfigxyo yr.rihzgryxibypo c.tbalyxigxy.hayzorp ygabycgugsnby.eypbgnordyfoxiyzgxbaognyr.xycabgxbpyslyxibzygs.tbygnnyxibyabgno gxo.ryxigxysbghxlysbn.rd yx.yxiby.hxbayf.anpyg yx.yxibyorrbaygabyxibyciobeyzbgr y.ey.tbac.zordyxibyxbaaosnby br by.eyozu.xbrcby.eyfbgkrb  y.eybvonbygzopyi. xonbyu.fba yfiociyo yx..yguxyx.yab hnxyea.zygckr.fnbpdordyxibygnnshxy.zrou.xbrcby.eygnobrye.acb jyx.yabc.rconbyh yslyxibybviosoxo.ry.eyox ygfehnysbghxlyx.yxibyabodry.eyegxbfiociyo yzbabnlyxibynoxbagalyuba .roeocgxo.ry.eyxib bye.acb o yxibyxg ky.eyxagdbpljyshxyzgxibzgxoc yxgkb yh y xonnyehaxibayea.zyfigxyo yihzgryorx.yxibyabdo.ry.eygs .nhxbyrbcb  oxlyx.yfiociyr.xy.rnlyxibygcxhgnyf.anpyshxybtbalyu.  osnbyf.anpyzh xyc.re.azygrpybtbryibabyoxyshonp ygyigsoxgxo.ry.ayagxibayeorp ygyigsoxgxo.rybxbargnnly xgrpordyfibaby.hayopbgn ygabyehnnly gxo eobpygrpy.haysb xyi.ub ygabyr.xyxifgaxbpjyoxyo y.rnlyfibryfbyxi.a.hdinlyhrpba xgrpyxibybrxoabyorpbubrpbrcby.ey.ha bntb yfiociysbn.rd yx.yxio yf.anpyxigxyabg .ryeorp yxigxyfbycgrygpbmhgxbnlyabgno byxibyua.e.hrpyozu.axgrcby.eyox ysbghxljyr.xy.rnlyo yzgxibzgxoc yorpbubrpbrxy.eyh ygrpy.hayxi.hdix yshxyorygr.xibay br byfbygrpyxibyfi.nbyhrotba by.eybvo xordyxiord ygabyorpbubrpbrxy.eyzgxibzgxoc jyxibyguuabibr o.ry.eyxio yuhabnlyopbgnycigagcxbayo yorpo ubr gsnbyoeyfbygabyx.yhrpba xgrpyaodixnlyxibyungcby.eyzgxibzgxoc yg y.rbygz.rdyxibygax jyoxyfg ye.azbanly huu. bpyxigxyuhabyabg .ryc.hnpypbcopbyory .zbyab ubcx yg yx.yxibyrgxhaby.eyxibygcxhgnyf.anpydb.zbxalygxynbg xyfg yxi.hdixyx.ypbgnyfoxiyxiby ugcbyoryfiociyfbynotbjyshxyfbyr.fykr.fyxigxyuhabyzgxibzgxoc ycgryrbtbayua.r.hrcbyhu.rymhb xo.r y.eygcxhgnybvo xbrcbyxibyf.anpy.eyabg .ryorygy br byc.rxa.n yxibyf.anpy.eyegcxyshxyoxyo yr.xygxygrlyu.orxycabgxotby.eyegcxygrpyoryxibyguunocgxo.ry.eyox yab hnx yx.yxibyf.anpyoryxozbygrpy ugcbyox ycbaxgorxlygrpyuabco o.rygabyn. xygz.rdyguua.vozgxo.r ygrpyf.akordyilu.xib b jyxiby.swbcx yc.r opbabpyslyzgxibzgxocogr yigtbyoryxibyug xysbbryzgornly.eygykorpy hddb xbpyslyuibr.zbrgyshxyea.zy hciyab xaocxo.r yxibygs xagcxyozgdorgxo.ry i.hnpysbyfi.nnlyeabbjygyabcoua.cgnynosbaxlyzh xyxih ysbygcc.apbpyabg .rycgrr.xypocxgxbyx.yxibyf.anpy.eyegcx yshxyxibyegcx ycgrr.xyab xaocxyabg .r yuaotonbdby.eypbgnordyfoxiyfigxbtbay.swbcx yox yn.tby.eysbghxlyzglycgh byx.y bbzyf.axily.eyc.r opbagxo.rjyibabyg ybn bfibabyfbyshonpyhuy.hay.fryopbgn y.hxy.eyxibyeagdzbrx yx.ysbye.hrpyoryxibyf.anpygrpyoryxibybrpyoxyo yigapyx.y glyfibxibayxibyab hnxyo ygycabgxo.ry.aygypo c.tbaljyoxyo ytbalypb oagsnbyoryor xahcxo.ryr.xyzbabnlyx.yuba hgpbyxiby xhpbrxy.eyxibygcchagcly.eyozu.axgrxyxib.abz yshxyx.yuba hgpbyiozyoryxibyfglyfiociyox bneyig y.eygnnyu.  osnbyfgl yxibyz. xysbghxljyxibyxahbyorxbab xy.eygypbz.r xagxo.ryo yr.xyg yxagpoxo.rgnyz.pb y.eybvu. oxo.ry hddb xyc.rcbrxagxbpyfi.nnlyoryxibyab hnxyfibabyxio yp.b y.cchayoxyzh xysbytobfbpyg ygypbebcxyx.ysbyabzbpobpyoeyu.  osnbysly .ydbrbagno ordyxiby xbu y.eyxibyua..eyxigxybgciysbc.zb yozu.axgrxyorygrpye.ayox bnejygrygadhzbrxyfiociy batb y.rnlyx.yua.tbygyc.rcnh o.ryo ynokbygy x.aly hs.aporgxbpyx.y .zbyz.agnyfiociyoxyo yzbgrxyx.yxbgciye.ay xibxocyubaebcxo.ryr.yugaxy.eyxibyfi.nby i.hnpysbyzbabnlygyzbgr jygycbaxgoryuagcxocgny uoaoxygypb oabye.ayaguopyua.dab  ye.ayc.rmhb xy.eyrbfyabgnz yo yab u.r osnbye.ayxibyhrphbybzuig o yhu.ryab hnx yfiociyuabtgon yoryzgxibzgxocgnyor xahcxo.rjyxibysbxxbayfglyo yx.yua.u. by .zbyxibzbye.ayc.r opbagxo.rorydb.zbxalygyeodhabyigtordyozu.axgrxyua.ubaxob yorygrgnl o ygyehrcxo.ry.eyfiociyxiby xhplyo yonnhzorgxordygrpy .y.rjyfibrbtbayua..e ypbubrpyhu.ry .zby.rnly.eyxibyzgak yslyfiociyfbypbeorbyxiby.swbcxyx.ysby xhpobpyxib byzgak y i.hnpysbyo .ngxbpygrpyortb xodgxbpy.ryxiboay.frygcc.hrxjye.ayoxyo ygypbebcxyorygrygadhzbrxyx.ybzun.lyz.abyuabzo  b yxigryxibyc.rcnh o.rypbzgrp yfigxyzgxibzgxocogr ycgnnybnbdgrcbyab hnx yea.zybzun.lordy.rnlyxibyb  brxognyuaorcounb yorytoaxhby.eyfiociyxibyxib o yo yxahbjyoxyo ygyzbaoxyorybhcnopyxigxyibygptgrcb yg yegayg yibyo ygsnbyx.yd.yfoxi.hxybzun.lordyxibygvo.zy.eyugagnnbn r.xyg yo y.exbry gopysbcgh byxio ygvo.zyo yoribabrxnly.swbcxo.rgsnbyshxysbcgh byoryzgxibzgxoc ybtbalyrbfygvo.zypozoro ib yxibydbrbagnoxly.eyxibyab hnxordyxib.abz ygrpyxibydabgxb xyu.  osnbydbrbagnoxlyo ysbe.abygnnyxiord yx.ysby .hdixjy.eyxibybeebcx y.eyzgxibzgxoc y.hx opbyox y.fry uibabyz.abyig ysbbryfaoxxbryxigry.ryxiby hswbcxy.eyox y.fryua.ubayopbgnjyxibybeebcxyhu.ryuion. .uilyig yoryxibyug xysbbryz. xyr.xgsnbyshxyz. xytgaobpyoryxiby btbrxbbrxiycbrxhalyopbgno zygrpyagxo.rgno zyoryxibybodixbbrxiyzgxbaogno zygrpy br gxo.rgno zy bbzbpybmhgnnlyox y.ee uaordjy.eyxibybeebcxyfiociyoxyo ynokbnlyx.yigtbyoryxibyehxhabyoxyf.hnpysbytbalyag iyx.y glyzhciyshxyory.rbyab ubcxygyd..pyab hnxyguubga yua.sgsnbjygdgor xyxigxykorpy.ey cbuxoco zyfiociygsgrp.r yxibyuha hoxy.eyopbgn ysbcgh byxibya.gpyo ygaph.h ygrpyxibyd.gnyr.xycbaxgornlygxxgorgsnbyzgxibzgxoc yfoxioryox y.fry uibabyo ygyc.zunbxbygr fbajyx..y.exbryoxyo y gopyxigxyxibabyo yr.ygs .nhxbyxahxiyshxy.rnly.uoro.rygrpyuaotgxbywhpdzbrxyxigxybgciy.eyh yo yc.rpoxo.rbpyoryio ytobfy.eyxibyf.anpyslyio y.fryubchnogaoxob yio y.fryxg xbygrpysog yxigxyxibabyo yr.ybvxbargnykordp.zy.eyxahxiyx.yfiociyslyugxobrcbygrpypo counorbyfbyzglygxyng xy.sxgorygpzoxxgrcbyshxy.rnlyxahxiye.ayzbye.ayl.hye.aybtbaly bugagxbyuba .rjyslyxio yigsoxy.eyzorpy.rby.eyxibyciobeybrp y.eyihzgrybee.axyo ypbrobpygrpyxiby huabzbytoaxhby.eycgrp.hay.eyebganb  ygckr.fnbpdzbrxy.eyfigxyo ypo guubga yea.zy.hayz.agnyto o.rjy.ey hciy cbuxoco zyzgxibzgxoc yo ygyubaubxhgnyabua..eye.ayox ybpoeocby.eyxahxi y xgrp yhr igkgsnbygrpyorbvuhrdgsnbyx.ygnnyxibyfbgu.r y.eyp.hsxordyclroco zjyxibybeebcx y.eyzgxibzgxoc yhu.ryuagcxocgnynoebyxi.hdiyxibly i.hnpyr.xysbyabdgapbpyg yxibyz.xotby.ey.hay xhpob yzglysbyh bpyx.ygr fbaygyp.hsxyx.yfiociyxiby .noxgaly xhpbrxyzh xygnfgl ysbynogsnbjyorygyf.anpy .yehnny.eybtonygrpy heebaordyabxoabzbrxyorx.yxibycn.o xbay.eyc.rxbzungxo.ryx.yxibybrw.lzbrxy.eypbnodix yfiociyi.fbtbayr.snbyzh xygnfgl ysbye.ayxibyebfy.rnlycgrr.xyshxyguubgayg ygy .zbfigxy bneo iyabeh gnyx.y igabyxibyshapbryozu. bpyhu.ry.xiba yslygccopbrx yoryfiociywh xocbyungl yr.yugaxjyigtbygrly.eyh yxibyaodixyfbyg kyx.yfoxipagfyea.zyuab brxybton yx.ynbgtby.hayebnn.fzbryhrgopbpyfionbyfbynotbygynoebyfiociyxi.hdiygaph.h ygrpygh xbabyo ylbxyungornlyd..pyoryox y.fryrgxhabyfibryxib bymhb xo.r ygao byxibyxahbygr fbayo yr.yp.hsxyxigxy .zbyzh xykbbuygnotbyxiby gcabpyeoaby .zbyzh xyuab batbyorybtbalydbrbagxo.ryxibyighrxordyto o.ryfiociy igp.f ye.axiyxibyd.gny.ey .yzhciy xaotordjyshxyfibryg yzh xy .zbxozb y.cchayxio ygr fbay bbz yx..yc.npyfibryfbygabygnz. xyzgppbrbpyslyxiby ubcxgcnby.ey .aa.f yx.yfiociyfbysaordyr.yibnuyxibryfbyzglyabenbcxyxigxyorpoabcxnlyxibyzgxibzgxocogry.exbryp.b yz.abye.ayihzgryiguuorb  yxigrygrly.eyio yz.abyuagcxocgnnlygcxotbyc.rxbzu.agaob jyxibyio x.aly.ey cobrcbygshrpgrxnlyua.tb yxigxygys.ply.eygs xagcxyua.u. oxo.r btbryoeyg yoryxibycg by.eyc.rocy bcxo.r yoxyabzgor yxf.yxi.h grpylbga yfoxi.hxybeebcxyhu.rypgonlynoebzglylbxygxygrlyz.zbrxysbyh bpyx.ycgh bygyabt.nhxo.ryoryxibyigsoxhgnyxi.hdix ygrpy.cchugxo.r y.eybtbalycoxoqbrjyxibyh by.ey xbgzygrpybnbcxaocoxlx.yxgkby xaokordyor xgrcb o yabrpbabpyu.  osnby.rnlyslyzgxibzgxoc jyoryxibyab hnx y.eygs xagcxyxi.hdixyxibyf.anpyu.  b  b ygycguoxgny.eyfiociyxibybzun.lzbrxyorybraociordyxibyc.zz.rya.hrpyig yr.yioxibax.ypo c.tbagsnbynozox jyr.ayp.b ybvubaobrcbydotbygrlyzbgr y.eypbcopordyfigxyugax y.eyzgxibzgxoc yfonnysbye.hrpyh behnjyhxonoxlyxibabe.abycgrysby.rnlygyc.r .ngxo.ryoryz.zbrx y.eypo c.hagdbzbrxyr.xygydhopbyorypoabcxordy.hay xhpob jye.ayxibyibgnxiy.eyxibyz.agnynoebye.aybrr.snordyxibyx.rby.eygrygdby.aygyrgxo.ryxibygh xbabaytoaxhb yigtbygy xagrdbyu.fbaybvcbbpordyxibyu.fbay.eyxi. byr.xyore.azbpygrpyuhaoeobpyslyxi.hdixjy.eyxib bygh xbabaytoaxhb yxibyn.tby.eyxahxiyo yxibyciobeygrpyoryzgxibzgxoc yz.abyxigrybn bfibabyxibyn.tby.eyxahxiyzglyeorpybrc.hagdbzbrxye.ayfgrordyegoxijybtbalydabgxy xhplyo yr.xy.rnlygrybrpyoryox bneyshxygn .ygyzbgr y.eycabgxordygrpy h xgorordygyn.exlyigsoxy.eyzorpygrpyxio yuhau. by i.hnpysbykbuxygnfgl yorytobfyxia.hdi.hxyxibyxbgciordygrpynbgarordy.eyzgxibzgxoc jy',
       'test.txt')
