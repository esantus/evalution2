package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class Definition2SynsetID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="synset_id")
	Long synset_id;
	
	@Column(name="definition_id")
	Long definition_id;
	
	@Column(name="language_id")
	Long language_id;

	
	public Definition2SynsetID() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Definition2SynsetID(Long synsetID, Long definition_id, Long languageID) {
		super();
		this.synset_id = synsetID;
		this.definition_id = definition_id;
		this.language_id = languageID;
	}

	public Long getSynset_id() {
		return synset_id;
	}

	public void setSynset_id(Long synset_id) {
		this.synset_id = synset_id;
	}

	public Long getDefinition_id() {
		return definition_id;
	}

	public void setDefinition_id(Long definition_id) {
		this.definition_id = definition_id;
	}

	public Long getLanguage_id() {
		return language_id;
	}

	public void setLanguage_id(Long language_id) {
		this.language_id = language_id;
	}
	
	@Override
	public String toString() {
		return "Defintion2SynsetID [synsetID=" + synset_id + ", definition_id=" + definition_id + ", languageID="
				+ language_id + "]";
	}


	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof Definition2SynsetID)) return false;
		Definition2SynsetID that = (Definition2SynsetID) o;
		return Objects.equals(getSynset_id(), that.getSynset_id()) && 
				Objects.equals(getDefinition_id(), that.getDefinition_id()) && 
				Objects.equals(getLanguage_id(), that.getLanguage_id());
	}
	
	public int hashCode(){
		return Objects.hash(getSynset_id(), getDefinition_id(), getLanguage_id());
	}
}
