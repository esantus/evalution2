package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="word2Synset")
public class Word2Synset{
	@EmbeddedId
	private Word2SynsetID id; 
	private Long wordSense_id;
	private Boolean mainWord;
	private Long pos_id;
	private Long wordType_id;
	private Double wordConcreteness;
	
	public Word2Synset() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Word2Synset(Word2SynsetID id, Long wordSense_id, Boolean mainWord, Long pos_id, Long wordType_id,
			Double wordConcreteness) {
		super();
		this.id = id;
		this.wordSense_id = wordSense_id;
		this.mainWord = mainWord;
		this.pos_id = pos_id;
		this.wordType_id = wordType_id;
		this.wordConcreteness = wordConcreteness;
	}

	public Word2SynsetID getId() {
		return id;
	}

	public void setId(Word2SynsetID id) {
		this.id = id;
	}

	public Long getWordSense_id() {
		return wordSense_id;
	}

	public void setWordSense_id(Long wordSense_id) {
		this.wordSense_id = wordSense_id;
	}

	public Boolean isMainWord() {
		return mainWord;
	}

	public void setMainWord(Boolean mainWord) {
		this.mainWord = mainWord;
	}

	public Long getPos_id() {
		return pos_id;
	}

	public void setPos_id(Long pos_id) {
		this.pos_id = pos_id;
	}

	public Long getWordType_id() {
		return wordType_id;
	}

	public void setWordType_id(Long wordType_id) {
		this.wordType_id = wordType_id;
	}

	public Double getWordConcreteness() {
		return wordConcreteness;
	}

	public void setWordConcreteness(Double wordConcreteness) {
		this.wordConcreteness = wordConcreteness;
	}


	@Override
	public String toString() {
		return "Word2Synset [id=" + id + ", wordSense_id=" + wordSense_id + ", mainWord=" + mainWord + ", pos_id="
				+ pos_id + ", wordType_id=" + wordType_id + ", wordConcreteness=" + wordConcreteness+ "]";
	}
}
