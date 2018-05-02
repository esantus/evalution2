package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class Word2SourceID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="synset_id")
	Long synset_id;
	
	@Column(name="word_id")
	Long word_id;

	@Column(name="language_id")
	Long language_id;

	@Column(name="source_id")
	Long source_id;
	
	
	public Word2SourceID() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Word2SourceID(Long synset_id, Long word_id, Long language_id, Long source_id) {
		super();
		this.synset_id = synset_id;
		this.word_id = word_id;
		this.language_id = language_id;
		this.source_id = source_id;
	}

	public Long getSynset_id() {
		return synset_id;
	}

	public void setSynset_id(Long synset_id) {
		this.synset_id = synset_id;
	}

	public Long getWord_id() {
		return word_id;
	}

	public void setWord_id(Long word_id) {
		this.word_id = word_id;
	}

	public Long getLanguage_id() {
		return language_id;
	}

	public void setLanguage_id(Long language_id) {
		this.language_id = language_id;
	}

	public Long getSource_id() {
		return source_id;
	}

	public void setSource_id(Long source_id) {
		this.source_id = source_id;
	}

	@Override
	public String toString() {
		return "Word2SourceID [synset_id=" + synset_id + ", word_id=" + word_id + ", language_id=" + language_id
				+ ", source_id=" + source_id + "]";
	}

	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof Word2SourceID)) return false;
		Word2SourceID that = (Word2SourceID) o;
		return Objects.equals(getSynset_id(), that.getSynset_id()) && 
				Objects.equals(getWord_id(), that.getWord_id()) &&
				Objects.equals(getLanguage_id(), that.getLanguage_id()) &&
				Objects.equals(getSource_id(), that.getSource_id());
	}
	
	public int hashCode(){
		return Objects.hash(getSynset_id(), getWord_id(), getLanguage_id(), getSource_id());
	}
}
