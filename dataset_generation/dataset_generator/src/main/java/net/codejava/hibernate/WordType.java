package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="wordType")
public class WordType {
	private Long wordType_id;
	private String wordType_value;
	
	
	public WordType() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public WordType(Long wordType_id, String wordType_value) {
		super();
		this.wordType_id = wordType_id;
		this.wordType_value = wordType_value;
	}
	
	@Id
	@Column(name = "wordType_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getWordType_id() {
		return wordType_id;
	}
	public void setWordType_id(Long wordType_id) {
		this.wordType_id = wordType_id;
	}

	public String getWordType_value() {
		return wordType_value;
	}
	
	public void setWordType_value(String wordType_value) {
		this.wordType_value = wordType_value;
	}

	@Override
	public String toString() {
		return "WordType [wordType_id=" + wordType_id + ", wordType_value=" + wordType_value + "]";
	}
}
