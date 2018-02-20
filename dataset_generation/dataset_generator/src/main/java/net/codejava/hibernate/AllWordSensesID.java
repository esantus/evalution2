package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class AllWordSensesID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="wordSense_id")
	Long wordSenses_id;
	
	@Column(name="word_id")
	Long word_id;
	
	@Column(name="language_id")
	Long language_id;

	
	public AllWordSensesID() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public AllWordSensesID(Long wordSenses_id, Long word_id, Long language_id) {
		super();
		this.wordSenses_id = wordSenses_id;
		this.word_id = word_id;
		this.language_id = language_id;
	}
	
	public Long getWordSenses_id() {
		return wordSenses_id;
	}

	public void setWordSenses_id(Long wordSenses_id) {
		this.wordSenses_id = wordSenses_id;
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

	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof AllWordSensesID)) return false;
		AllWordSensesID that = (AllWordSensesID) o;
		return Objects.equals(getWordSenses_id(), that.getWordSenses_id()) && 
				Objects.equals(getWord_id(), that.getWord_id()) && 
				Objects.equals(getLanguage_id(), that.getLanguage_id());
	}
	
	public int hashCode(){
		return Objects.hash(getWordSenses_id(), getWord_id(), getLanguage_id());
	}
}
