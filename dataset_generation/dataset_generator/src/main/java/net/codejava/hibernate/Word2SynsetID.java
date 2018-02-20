package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class Word2SynsetID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="synset_id")
	Long synsetID;
	
	@Column(name="word_id")
	Long wordID;
	
	@Column(name="language_id")
	Long languageID;

	
	public Word2SynsetID() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public Word2SynsetID(Long synsetID, Long wordID, Long languageID) {
		super();
		this.synsetID = synsetID;
		this.wordID = wordID;
		this.languageID = languageID;
	}


	public Long getSynsetID() {
		return synsetID;
	}


	public void setSynsetID(Long synsetID) {
		this.synsetID = synsetID;
	}


	public Long getWordID() {
		return wordID;
	}


	public void setWordID(Long wordID) {
		this.wordID = wordID;
	}


	public Long getLanguageID() {
		return languageID;
	}


	public void setLanguageID(Long languageID) {
		this.languageID = languageID;
	} 
	
	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof Word2SynsetID)) return false;
		Word2SynsetID that = (Word2SynsetID) o;
		return Objects.equals(getSynsetID(), that.getSynsetID()) && 
				Objects.equals(getWordID(), that.getWordID()) && 
				Objects.equals(getLanguageID(), that.getLanguageID());
	}
	
	public int hashCode(){
		return Objects.hash(getSynsetID(), getWordID(), getLanguageID());
	}
}
