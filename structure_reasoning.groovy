@Grapes([
    @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5'),
    @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3'),
    @GrabConfig(systemClassLoader=true)
])

import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.*
import org.semanticweb.elk.owlapi.ElkReasonerFactory
import org.semanticweb.owlapi.reasoner.*

import java.io.*

def ontologyPath = "./Data/hp(updated).owl"
def outputPath = "./Data/structure.txt"

OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ontology = manager.loadOntologyFromOntologyDocument(new File(ontologyPath))

OWLReasonerFactory reasonerFactory = new ElkReasonerFactory()
OWLReasoner reasoner = reasonerFactory.createReasoner(ontology)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)

def writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputPath), "UTF-8"))
ontology.getClassesInSignature().each { cls ->
    def classIRI = "<" + cls.getIRI().getShortForm() + ">"

    def superNodes = reasoner.getSuperClasses(cls, false)
    superNodes.getFlattened().each { sup ->
        def superIRI = "<" + sup.getIRI().getShortForm() + ">"
        writer.println("${classIRI} SubClassOf ${superIRI}")
    }
}

writer.close()
println "Reasoning done：${outputPath}"

