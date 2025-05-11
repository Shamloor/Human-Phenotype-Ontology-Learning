@Grapes([
    @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5'),
    @GrabConfig(systemClassLoader=true)
])

import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.*

import java.io.*

def ontologyPath = "./Data/owl/hp(original).owl"
def outputPath = "./Data/embedding/structure.txt"

OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ontology = manager.loadOntologyFromOntologyDocument(new File(ontologyPath))

def writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputPath), "UTF-8"))

def hasSuperclass = new HashSet<OWLClass>()

ontology.getAxioms(AxiomType.SUBCLASS_OF).each { axiom ->
    def sub = axiom.getSubClass()
    def sup = axiom.getSuperClass()
    if (sub instanceof OWLClass && sup instanceof OWLClass) {
        hasSuperclass.add(sub)
        def subIRI = "<" + sub.getIRI().getShortForm() + ">"
        def supIRI = "<" + sup.getIRI().getShortForm() + ">"
        writer.println("${subIRI} ${supIRI}")
    }
}

ontology.getClassesInSignature(true).each { cls ->
    if (!cls.isOWLThing() && !cls.isOWLNothing() && !hasSuperclass.contains(cls)) {
        def classIRI = "<" + cls.getIRI().getShortForm() + ">"
        writer.println("${classIRI} <Thing>")
    }
}

writer.close()
println "Structure extraction completed with root-level <Thing> links: ${outputPath}"