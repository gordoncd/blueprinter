# blueprinter
Make a electrical power grid component blueprint by giving textual prompt with specifications including component, from_kv, to_kv, rating_mva, scheme_hv, scheme_lv, lv_feeders.


Project specifications: 

input: a text prompt with specifications about local transformer substation
output: a blueprint of the local transformer substation in single line diagram format

The input is converted to an intent file (intent.json) which is then processed by an action generator. So it goes query -> intent.json 

There is a diagram file which can programmatically generate the blueprint json of specifications from the action generator output. So it goes intent.json -> action generator -> diagram.json. The diagram.json file contains the entities, their ids, and their connections in the form of connection objects (bus connections and couplers and bays)

then a blueprint maker deterministically builds the blueprint from the diagram.json file. So it goes diagram.json -> blueprint single line diagram. 
