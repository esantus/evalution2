package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="word2Source")
public class Word2Source{
	@EmbeddedId
	private Word2SourceID id;
	
	public Word2Source() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Word2Source(Word2SourceID id) {
		super();
		this.id = id;
	}

	public Word2SourceID getId() {
		return id;
	}

	public void setId(Word2SourceID id) {
		this.id = id;
	}

	@Override
	public String toString() {
		return "Word2Source [id=" + id + "]";
	}
	
	
}
