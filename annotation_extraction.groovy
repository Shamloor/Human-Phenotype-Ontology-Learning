@Grapes([
    @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5'),
    @GrabConfig(systemClassLoader=true)
])

import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.*
import org.semanticweb.owlapi.search.EntitySearcher

import java.io.*

def ontologyPath = "./Data/owl/hp(updated).owl"
def outputPath = "./Data/annotation.txt"

def manager = OWLManager.createOWLOntologyManager()
def ontology = manager.loadOntologyFromOntologyDocument(new File(ontologyPath))
def factory = manager.getOWLDataFactory()

def targetProperties = [
    "rdfs:label"         : factory.getRDFSLabel(),
    "IAO_0000115"        : factory.getOWLAnnotationProperty(IRI.create("http://purl.obolibrary.org/obo/IAO_0000115")),
    "hasExactSynonym"    : factory.getOWLAnnotationProperty(IRI.create("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")),
    "hasRelatedSynonym"  : factory.getOWLAnnotationProperty(IRI.create("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym")),
    "hasNarrowSynonym"   : factory.getOWLAnnotationProperty(IRI.create("http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym"))
]

def writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputPath), "UTF-8"))

ontology.getClassesInSignature().each { cls ->
    def classIRI = "<" + cls.getIRI().getShortForm() + ">"
    targetProperties.each { label, prop ->
        EntitySearcher.getAnnotations(cls, ontology, prop).each { annotation ->
            def val = annotation.getValue()
            if (val instanceof OWLLiteral) {
                def literal = val.getLiteral().replaceAll("\\s+", " ").trim()
                writer.println("${classIRI} ${label} ${literal}")
            }
        }
    }
}

writer.close()
println "Annotation done：${outputPath}"
