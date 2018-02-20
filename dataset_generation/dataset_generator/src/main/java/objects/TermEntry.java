package objects;


public class TermEntry {
	
	String entryID;
	String originalEntryID;
	String language;
	String term;
	String domain;
	String definition;
	String source;
	String termType;
	String reliability;
	
	public TermEntry() {
		super();
		// TODO Auto-generated constructor stub
	}

	public TermEntry(String entryID, String originalEntryID, String language, String term, String domain,
			String definition, String source, String termType, String reliability) {
		super();
		this.entryID = entryID;
		this.originalEntryID = originalEntryID;
		this.language = language;
		this.term = term;
		this.domain = domain;
		this.definition = definition;
		this.source = source;
		this.termType = termType;
		this.reliability = reliability;
	}
	
	public String getEntryID() {
		return entryID;
	}
	public void setEntryID(String entryID) {
		this.entryID = entryID;
	}	
	public String getOriginalEntryID() {
		return originalEntryID;
	}

	public void setOriginalEntryID(String originalEntryID) {
		this.originalEntryID = originalEntryID;
	}
	public String getLanguage() {
		return language;
	}
	public void setLanguage(String language) {
		this.language = language;
	}
	public String getTerm() {
		return term;
	}
	public void setTerm(String term) {
		this.term = term;
	}
	public String getDomain() {
		return domain;
	}
	public void setDomain(String domain) {
		this.domain = domain;
	}
	public String getDefinition() {
		return definition;
	}
	public void setDefinition(String definition) {
		this.definition = definition;
	}
	public String getSource() {
		return source;
	}
	public void setSource(String source) {
		this.source = source;
	}
	public String getTermType() {
		return termType;
	}
	public void setTermType(String termType) {
		this.termType = termType;
	}
	public String getReliability() {
		return reliability;
	}
	public void setReliability(String reliability) {
		this.reliability = reliability;
	}
	
	@Override
	public String toString() {
		return "TermEntry [entryID=" + entryID + ", originalEntryID=" + originalEntryID + ", language=" + language
				+ ", term=" + term + ", domain=" + domain + ", definition=" + definition + ", source=" + source
				+ ", termType=" + termType + ", reliability=" + reliability + "]";
	}

	@Override
	public int hashCode() {
	    return (getEntryID().hashCode() + getLanguage().hashCode()+ getTerm().hashCode());
	}
	
	 @Override
	 public boolean equals (Object object) {
	     boolean result = false;
	     if (object == null || object.getClass() != getClass()) {
	         result = false;
	     } else {
	         TermEntry relatum = (TermEntry) object;
	         if (this.entryID.equals(relatum.getEntryID()) && this.language.equals(relatum.getLanguage()) && this.term.equals(relatum.getTerm())) {
	             result = true;
	         }
	     }
	     return result;
	 }
}
