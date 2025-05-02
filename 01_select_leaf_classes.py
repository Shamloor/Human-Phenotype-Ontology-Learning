from rdflib import Graph

g = Graph()
g.parse("Data/hp.owl", format='xml')

query_for_class = """
SELECT DISTINCT ?class
WHERE {
  ?class a owl:Class .
  FILTER(isIRI(?class))

  FILTER NOT EXISTS {
    ?subclass rdfs:subClassOf ?class .
  }
}
"""

results_of_class = g.query(query_for_class)

class_uris = [str(row[0]) for row in results_of_class]

output_path = "Data/leaf classes.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for uri in class_uris:
        f.write(uri + "\n")

print(f"保存完成，共 {len(class_uris)} 个class，文件路径：{output_path}")




