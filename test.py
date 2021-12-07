import numpy as np
import csv
a = [('immensely', 0.8415359258651733), ('provision', 0.7728386521339417), ('haphazard', 0.5940636992454529),
            ('influential', 0.6994330883026123), ('apogee', 0.7258920669555664), ('deeply_flawed', 0.7917903661727905),
            ('urgent', 0.7307383418083191), ('consume', 0.6981646418571472), ('introduction', 0.5692579746246338), 
            ('succinctly', 0.5996754765510559), ('salute', 0.7670279145240784), ('lonely', 0.5398979783058167), 
            ('hastened', 0.6957820653915405), ('tenacity', 0.7534494400024414), ('fantastical', 0.6945845484733582), 
            ('showing', 0.753765344619751), ('continually', 0.8584363460540771), ('issue', 0.7072023153305054), ('furnished', 0.647893488407135), 
            ('expensive', 0.7346715331077576), ('Recognized', 0.6635153889656067), ('spots', 0.7677314877510071), ('making', 0.7594648003578186), 
            ('sometimes', 0.8321627378463745), ('affable', 0.7319809198379517), ('debates', 0.8205613493919373), ('Narrow', 0.6413078904151917), 
            ('arrange', 0.7053579092025757), ('limitless', 0.7255815267562866), ('flashy', 0.6279592514038086), ('levying', 0.7044811844825745), 
            ('skillfully', 0.8181504607200623), ('distributing', 0.7066061496734619), ('discrepancy', 0.8063086867332458), 
            ('Prolific', 0.6275216937065125), ('unparalleled', 0.8846917748451233), ('ineffably', 0.6069270372390747), 
            ('hues', 0.6374392509460449), ('hindquarters', 0.6097269654273987), ('highlights', 0.7327486872673035), ('hurriedly', 0.7321720719337463), 
            ('temperate_climate', 0.6393973231315613), ('smile', 0.860400915145874), ('verbal', 0.631680428981781), ('doctor', 0.7806021571159363), 
            ('basically', 0.8192141056060791), ('eager', 0.6450806856155396), ('located', 0.8085098266601562), ('Principal', 0.7120686173439026), 
            ('gradually', 0.7653231620788574), ('constructed', 0.7734819650650024), ('mundane_tasks', 0.7368724942207336), ('likely', 0.7496002316474915), 
            ('half_heartedly', 0.5789012908935547), ('history', 0.6059836745262146), ('crazily', 0.6416054964065552), ('lauded', 0.7858651876449585), 
            ('commands', 0.5995240807533264), ('concocting', 0.7313812971115112), ('Prospective', 0.6062800884246826), ('typically', 0.7665401697158813), 
            ('sustaining', 0.7001972198486328), ('precarious', 0.7064183950424194), ('tranquillity', 'Word does not Exist', 0), 
            ('dissipated', 0.7107398509979248), ('principally', 0.8755374550819397), ('slang', 0.686839759349823), ('solved', 0.7185725569725037), 
            ('economically_feasible', 0.7941259741783142), ('expeditious_manner', 0.6898871660232544), ('per_centage', 0.7483636736869812), 
            ('terminate', 0.7549267411231995), ('uniforms', 0.714779257774353), ('figures', 0.5928177833557129), ('adequate', 0.7405216097831726), 
            ('fashions', 0.6891328692436218), ('Marketed', 0.6433728933334351), ('larger', 0.7976310849189758), ('origins', 0.6270325183868408), 
            ('usually', 0.8023927211761475)]


b = [] 
for i in a:
    b.append(i[0])

print(b)
print('/n')
l = [(b),(b)]
l1 = list(zip(*l))

print (l1)


# with open('test.csv', 'w', newline='') as myfile:
#      wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#      wr.writerow(b)